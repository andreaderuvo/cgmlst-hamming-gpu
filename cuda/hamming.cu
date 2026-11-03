// Hamming distance on the GPU, with two interchangeable per-locus operations:
//
//  * compare (default, fastest): [a != b && a != 0 && b != 0];
//  * relu (rh_run(..., relu=1, ...)): the same bit computed by a hand-wired
//    ReLU network, ~2.3x slower because it costs more arithmetic.
//
// In the ReLU mode each locus pair (a, b) — dense per-locus allele ranks, 0 = missing — goes
// through a fixed 3-layer ReLU network whose output is exactly
// [a != b and a > 0 and b > 0]:
//
//   diff = relu(a-b) + relu(b-a) - relu(a-b-1) - relu(b-a-1)   = min(1, |a-b|)
//   pa   = a - relu(a-1)                                         = min(1, a)
//   pb   = b - relu(b-1)                                         = min(1, b)
//   out  = relu(diff + pa + pb - 2)                              = AND of the three
//
// Exact for integers; ranks are < 2^16 so every value is exact in fp32. The
// distance is the sum of `out` over loci. The kernel is tiled like a GEMM
// (shared-memory tiles over loci, a TM x TN register block per thread) with
// the multiply-add replaced by the network. pa/pb depend on one operand only,
// so they are evaluated once per tile element instead of once per pair.
// Both modes give identical results.
//
// The matrix is produced in row stripes (only tiles on or above the diagonal),
// double-buffered on two streams: while the host consumes stripe s, the GPU
// computes stripe s+1. Stripe height adapts to the free VRAM.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define THREADS_X 16
#define THREADS_Y 16
#define TM 4
#define TN 4
#define BK 16
#define BM (THREADS_Y * TM)
#define BN (THREADS_X * TN)

__device__ __forceinline__ float relu(float z) { return fmaxf(z, 0.f); }

template <bool RELU>
__global__ void __launch_bounds__(THREADS_X * THREADS_Y)
hamming_stripe(const uint16_t* __restrict__ X, int n, int L, int r0, int rows,
                    uint16_t* __restrict__ out) {
    const int row0 = r0 + blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;
    if (col0 + BN <= row0) return;  // tile entirely below the diagonal

    __shared__ float As[BK][BM + 1];
    __shared__ float Bs[BK][BN + 1];
    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * THREADS_X + tx;
    const int rend = r0 + rows;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < L; k0 += BK) {
        for (int e = tid; e < BM * BK; e += THREADS_X * THREADS_Y) {
            int r = e / BK, kk = e % BK, gk = k0 + kk;
            int gr = row0 + r, gc = col0 + r;
            As[kk][r] = (gr < rend && gk < L) ? (float)X[(size_t)gr * L + gk] : 0.f;
            Bs[kk][r] = (gc < n && gk < L) ? (float)X[(size_t)gc * L + gk] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float a[TM], pa[TM], b[TN], pb[TN];
#pragma unroll
            for (int i = 0; i < TM; i++) {
                a[i] = As[kk][ty + THREADS_Y * i];
                pa[i] = RELU ? a[i] - relu(a[i] - 1.f) : 0.f;
            }
#pragma unroll
            for (int j = 0; j < TN; j++) {
                b[j] = Bs[kk][tx + THREADS_X * j];
                pb[j] = RELU ? b[j] - relu(b[j] - 1.f) : 0.f;
            }
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) {
                    if (RELU) {
                        float d = a[i] - b[j];
                        float diff = relu(d) + relu(-d) - relu(d - 1.f) - relu(-d - 1.f);
                        acc[i][j] += relu(diff + pa[i] + pb[j] - 2.f);
                    } else {
                        acc[i][j] += (a[i] != b[j] && a[i] != 0.f && b[j] != 0.f) ? 1.f : 0.f;
                    }
                }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; i++) {
        int gr = row0 + ty + THREADS_Y * i;
        if (gr >= rend) continue;
#pragma unroll
        for (int j = 0; j < TN; j++) {
            int gc = col0 + tx + THREADS_X * j;
            if (gc < n) out[(size_t)(gr - r0) * n + gc] = (uint16_t)acc[i][j];
        }
    }
}

typedef void (*rh_stripe_cb)(void* ctx, int r0, int rows, const uint16_t* stripe);

#define CK(x)                                                                         \
    do {                                                                              \
        cudaError_t e_ = (x);                                                         \
        if (e_ != cudaSuccess) {                                                      \
            snprintf(err, errlen, "%s: %s", #x, cudaGetErrorString(e_));              \
            goto fail;                                                                \
        }                                                                             \
    } while (0)

extern "C" int rh_device_count(void) {
    int c = 0;
    if (cudaGetDeviceCount(&c) != cudaSuccess) return 0;
    return c;
}

// X: n x L row-major ranks. Calls cb(ctx, r0, rows, stripe) for every stripe in
// order; stripe is rows x n (row-major), valid for columns >= the row index.
// `stripe_rows` 0 = auto; `relu` != 0 selects the ReLU-network kernel. Returns 0 on success, writes the message into err.
extern "C" int rh_run(const uint16_t* X, int n, int L, int device, int stripe_rows, int relu,
                      rh_stripe_cb cb, void* ctx, double* kernel_secs, char* err, int errlen) {
    uint16_t* dX = NULL;
    uint16_t* dO[2] = {NULL, NULL};
    uint16_t* hO[2] = {NULL, NULL};
    cudaStream_t st[2] = {NULL, NULL};
    cudaEvent_t done[2] = {NULL, NULL}, t0 = NULL, t1 = NULL;
    int S = stripe_rows, ns = 0, rc = 1;
    size_t freeb = 0, totb = 0, row_bytes = (size_t)n * sizeof(uint16_t);
    float ms = 0.f;

    CK(cudaSetDevice(device));
    CK(cudaMalloc(&dX, (size_t)n * L * sizeof(uint16_t)));
    CK(cudaMemcpy(dX, X, (size_t)n * L * sizeof(uint16_t), cudaMemcpyHostToDevice));
    if (S <= 0) {  // two device stripe buffers within half of the remaining VRAM
        CK(cudaMemGetInfo(&freeb, &totb));
        size_t s = freeb / 2 / (2 * row_bytes);
        s = s / BM * BM;
        S = (int)(s < BM ? BM : (s > 8192 ? 8192 : s));
    }
    if (S > n) S = (n + BM - 1) / BM * BM;
    for (int b = 0; b < 2; b++) {
        CK(cudaMalloc(&dO[b], (size_t)S * row_bytes));
        CK(cudaMallocHost(&hO[b], (size_t)S * row_bytes));
        CK(cudaStreamCreateWithFlags(&st[b], cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&done[b], cudaEventDisableTiming));
    }
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));

    ns = (n + S - 1) / S;
    CK(cudaEventRecord(t0, st[0]));
    for (int s = 0; s <= ns; s++) {
        if (s < ns) {
            int b = s & 1, r0 = s * S, rows = (n - r0 < S) ? n - r0 : S;
            dim3 grid((n + BN - 1) / BN, (rows + BM - 1) / BM), block(THREADS_X, THREADS_Y);
            if (relu)
                hamming_stripe<true><<<grid, block, 0, st[b]>>>(dX, n, L, r0, rows, dO[b]);
            else
                hamming_stripe<false><<<grid, block, 0, st[b]>>>(dX, n, L, r0, rows, dO[b]);
            CK(cudaGetLastError());
            CK(cudaMemcpyAsync(hO[b], dO[b], (size_t)rows * row_bytes, cudaMemcpyDeviceToHost, st[b]));
            CK(cudaEventRecord(done[b], st[b]));
        }
        if (s > 0) {  // consume the previous stripe while this one runs
            int p = (s - 1) & 1, r0 = (s - 1) * S, rows = (n - r0 < S) ? n - r0 : S;
            CK(cudaEventSynchronize(done[p]));
            cb(ctx, r0, rows, hO[p]);
        }
    }
    CK(cudaEventRecord(t1, st[(ns - 1) & 1]));
    CK(cudaEventSynchronize(t1));
    CK(cudaEventElapsedTime(&ms, t0, t1));
    if (kernel_secs) *kernel_secs = ms / 1000.0;
    rc = 0;
fail:
    for (int b = 0; b < 2; b++) {
        if (dO[b]) cudaFree(dO[b]);
        if (hO[b]) cudaFreeHost(hO[b]);
        if (st[b]) cudaStreamDestroy(st[b]);
        if (done[b]) cudaEventDestroy(done[b]);
    }
    if (t0) cudaEventDestroy(t0);
    if (t1) cudaEventDestroy(t1);
    if (dX) cudaFree(dX);
    return rc;
}
