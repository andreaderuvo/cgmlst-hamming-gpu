// GPU side of tc-hamming: exact cgMLST Hamming on 1-bit tensor cores.
//
//   H = P Pᵀ − O Oᵀ − R
//   P: n × L presence bits, O: n × D one-hot bits of frequent alleles (both
//   bit-packed row-major, rows padded with zeros), R: equal pairs of rare
//   alleles (CSR by row: start[i]..start[i+1] in js, partners j > i).
//
// Both products are AND+popcount GEMMs with int32 accumulation (exact); the
// second GEMM's epilogue computes P Pᵀ − O Oᵀ in place; rare pairs are
// subtracted with atomics (on the host if they do not fit in VRAM). Only the
// upper triangle, in row stripes, on two streams: while the host consumes
// stripe s (u16, rows × (n − r0), columns r0..n−1), stripe s+1 is computed.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <algorithm>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

using Gemm = cutlass::gemm::device::Gemm<
    cutlass::uint1b_t, cutlass::layout::RowMajor,
    cutlass::uint1b_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor, int32_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 512>,
    cutlass::gemm::GemmShape<64, 64, 512>,
    cutlass::gemm::GemmShape<16, 8, 256>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 128, 128,
    false, cutlass::arch::OpAndPopc>;

static size_t up(size_t x, size_t m) { return (x + m - 1) / m * m; }

__global__ void sub_rare(int32_t* out, size_t ldd, int r0, int rows,
                         const uint32_t* start, const uint32_t* js) {
    int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (r >= rows) return;
    int i = r0 + r;
    int32_t* row = out + (size_t)r * ldd - r0;
    for (uint32_t p = start[i] + lane; p < start[i + 1]; p += 32) atomicSub(row + js[p], 1);
}

__global__ void pack_u16(const int32_t* __restrict__ in, size_t ldd, uint16_t* __restrict__ out,
                         size_t w, int rows) {
    size_t total = (size_t)rows * w;
    for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < total; e += (size_t)gridDim.x * blockDim.x)
        out[e] = (uint16_t)in[(e / w) * ldd + e % w];
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    snprintf(err, errlen, "%s: %s", #x, cudaGetErrorString(e_)); goto fail; } } while (0)

static int gemm(int M, int N, int K, const uint8_t* A, int ldk, int32_t* D, int ldd,
                int beta, cudaStream_t s, char* err, int errlen) {
    Gemm op;
    Gemm::Arguments args({M, N, K}, {(cutlass::uint1b_t*)A, ldk}, {(cutlass::uint1b_t*)A, ldk},
                         {D, ldd}, {D, ldd}, {1, beta});
    cutlass::Status st = op(args, nullptr, s);
    if (st != cutlass::Status::kSuccess) {
        snprintf(err, errlen, "CUTLASS %s (M=%d N=%d K=%d)", cutlassGetStatusString(st), M, N, K);
        return 1;
    }
    return 0;
}

extern "C" int tch_device_count(void) {
    int c = 0;
    return cudaGetDeviceCount(&c) == cudaSuccess ? c : 0;
}

extern "C" size_t tch_free_vram(int device) {
    size_t f = 0, t = 0;
    if (cudaSetDevice(device) != cudaSuccess || cudaMemGetInfo(&f, &t) != cudaSuccess) return 0;
    return f;
}

typedef void (*tch_cb)(void* ctx, int r0, int rows, size_t w, const uint16_t* stripe);

// O: n_pad × K1 bits, P: n_pad × K2 bits (K1, K2 multiples of 1024, n_pad of 256).
extern "C" int tch_run(int device, int n, size_t n_pad,
                       const uint8_t* O, size_t K1, const uint8_t* P, size_t K2,
                       const uint32_t* start, const uint32_t* js, size_t n_rare,
                       int S, tch_cb cb, void* ctx, double* gpu_secs, int* rare_on_gpu,
                       char* err, int errlen) {
    uint8_t *dO = NULL, *dP = NULL;
    uint32_t *dStart = NULL, *dJ = NULL;
    int32_t* dH[2] = {NULL, NULL};
    uint16_t *dU[2] = {NULL, NULL}, *hU[2] = {NULL, NULL};
    cudaStream_t st[2] = {NULL, NULL};
    cudaEvent_t done[2] = {NULL, NULL}, t0 = NULL, t1 = NULL;
    const size_t ldd = n_pad;
    size_t freeb = 0, totb = 0;
    bool rare_gpu = false;
    int rc = 1, prev = -1;
    float ms = 0.f;

    CK(cudaSetDevice(device));
    CK(cudaMalloc(&dO, n_pad * K1 / 8));
    CK(cudaMalloc(&dP, n_pad * K2 / 8));
    CK(cudaMemcpy(dO, O, n_pad * K1 / 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dP, P, n_pad * K2 / 8, cudaMemcpyHostToDevice));
    CK(cudaMemGetInfo(&freeb, &totb));
    rare_gpu = (n_rare + n + 1) * 4 + (size_t)S * ldd * 12 + (64u << 20) < freeb;
    if (rare_gpu) {
        CK(cudaMalloc(&dStart, (size_t)(n + 1) * 4));
        CK(cudaMalloc(&dJ, std::max<size_t>(n_rare, 1) * 4));
        CK(cudaMemcpy(dStart, start, (size_t)(n + 1) * 4, cudaMemcpyHostToDevice));
        if (n_rare) CK(cudaMemcpy(dJ, js, n_rare * 4, cudaMemcpyHostToDevice));
    }
    *rare_on_gpu = rare_gpu;
    for (int b = 0; b < 2; b++) {
        CK(cudaMalloc(&dH[b], (size_t)S * ldd * 4));
        CK(cudaMalloc(&dU[b], (size_t)S * ldd * 2));
        CK(cudaMallocHost(&hU[b], (size_t)S * ldd * 2));
        CK(cudaStreamCreateWithFlags(&st[b], cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&done[b], cudaEventDisableTiming));
    }
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0, st[0]));
    for (int r0 = 0;; r0 += S) {
        if (r0 < n) {
            const int rows = std::min(S, n - r0), b = (r0 / S) & 1;
            const int M = (int)up(rows, 8), N = (int)up(n - r0, 8);
            const size_t w = n - r0;
            if (gemm(M, N, (int)K1, dO + (size_t)r0 * K1 / 8, (int)K1, dH[b], (int)ldd, 0, st[b], err, errlen)) goto fail;
            if (gemm(M, N, (int)K2, dP + (size_t)r0 * K2 / 8, (int)K2, dH[b], (int)ldd, -1, st[b], err, errlen)) goto fail;
            if (rare_gpu) sub_rare<<<(rows + 7) / 8, 256, 0, st[b]>>>(dH[b], ldd, r0, rows, dStart, dJ);
            pack_u16<<<1024, 256, 0, st[b]>>>(dH[b], ldd, dU[b], w, rows);
            CK(cudaGetLastError());
            CK(cudaMemcpyAsync(hU[b], dU[b], (size_t)rows * w * 2, cudaMemcpyDeviceToHost, st[b]));
            CK(cudaEventRecord(done[b], st[b]));
        }
        if (prev >= 0) {  // consume the previous stripe while this one runs
            const int rows = std::min(S, n - prev), b = (prev / S) & 1;
            const size_t w = n - prev;
            CK(cudaEventSynchronize(done[b]));
            if (!rare_gpu)
                for (int r = 0; r < rows; r++) {
                    const int i = prev + r;
                    for (uint32_t p = start[i]; p < start[i + 1]; p++) hU[b][(size_t)r * w + (js[p] - prev)] -= 1;
                }
            cb(ctx, prev, rows, w, hU[b]);
        }
        if (r0 >= n) break;
        prev = r0;
    }
    CK(cudaEventRecord(t1, st[0]));
    CK(cudaEventSynchronize(t1));
    CK(cudaEventElapsedTime(&ms, t0, t1));
    if (gpu_secs) *gpu_secs = ms / 1e3;
    rc = 0;
fail:
    for (int b = 0; b < 2; b++) {
        if (dH[b]) cudaFree(dH[b]);
        if (dU[b]) cudaFree(dU[b]);
        if (hU[b]) cudaFreeHost(hU[b]);
        if (st[b]) cudaStreamDestroy(st[b]);
        if (done[b]) cudaEventDestroy(done[b]);
    }
    if (t0) cudaEventDestroy(t0);
    if (t1) cudaEventDestroy(t1);
    if (dO) cudaFree(dO);
    if (dP) cudaFree(dP);
    if (dStart) cudaFree(dStart);
    if (dJ) cudaFree(dJ);
    return rc;
}
