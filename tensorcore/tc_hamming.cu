// Exact cgMLST Hamming distance on 1-bit tensor cores.
//
//   H(i,j) = C(i,j) - E(i,j)
//   C = #loci called in both          = P P^T      (P: n x L presence bits)
//   E = #loci called in both and equal = O O^T + R  (O: one-hot bits of alleles seen >= tau times,
//                                                    R: equal pairs of rarer alleles, added on the host)
//
// Both products are AND+popcount GEMMs (CUTLASS, int32 accumulation: exact). The
// second GEMM's epilogue computes P P^T - (O O^T) in place. Only the upper
// triangle is computed, in row stripes; rare pairs are subtracted per stripe.
//
//   ./tc_hamming <ranks file> [tau=0 auto] [stripe rows=512] [noverify]
// ranks file: int32 n, int32 L, then n*L uint16 dense per-locus ranks (0 = missing).
// Prints timings and the upper-triangle checksum used by bench_file.cu.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <random>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

using Gemm = cutlass::gemm::device::Gemm<
    cutlass::uint1b_t, cutlass::layout::RowMajor,
    cutlass::uint1b_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor, int32_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>,
    cutlass::gemm::GemmShape<W_M, W_N, TB_K>,
    cutlass::gemm::GemmShape<16, 8, 256>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, STAGES, 128, 128,
    false, cutlass::arch::OpAndPopc>;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static size_t up(size_t x, size_t m) { return (x + m - 1) / m * m; }

// Rare equal pairs: one warp per stripe row, lanes stride over that row's
// partners (a pair can repeat across loci, hence the atomic).
__global__ void sub_rare(int32_t* out, size_t ldd, int r0, int rows,
                         const uint32_t* start, const uint32_t* js) {
    int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (r >= rows) return;
    int i = r0 + r;
    int32_t* row = out + (size_t)r * ldd - r0;
    for (uint32_t p = start[i] + lane; p < start[i + 1]; p += 32) atomicSub(row + js[p], 1);
}

// int32 stripe -> u16 (distances <= L < 65536), packed with row width w for the copy-back.
__global__ void pack_u16(const int32_t* __restrict__ in, size_t ldd, uint16_t* __restrict__ out,
                         size_t w, int rows) {
    size_t total = (size_t)rows * w;
    for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < total; e += (size_t)gridDim.x * blockDim.x)
        out[e] = (uint16_t)in[(e / w) * ldd + e % w];
}

static void gemm(int M, int N, int K, const uint8_t* A, const uint8_t* B, int ldk,
                 const int32_t* C, int32_t* D, int ldd, int alpha, int beta, cudaStream_t stream) {
    Gemm op;
    Gemm::Arguments args({M, N, K},
                         {(cutlass::uint1b_t*)A, ldk}, {(cutlass::uint1b_t*)B, ldk},
                         {(int32_t*)C, ldd}, {D, ldd}, {alpha, beta});
    cutlass::Status st = op(args, nullptr, stream);
    if (st != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS error %s (M=%d N=%d K=%d)\n", cutlassGetStatusString(st), M, N, K);
        exit(1);
    }
}

// Pick tau minimising  GEMM time (n^2/2 * K1 bits at the measured L4 rate) + rare-pair time,
// subject to: one-hot bits + rare pairs + stripe buffers fitting in the free VRAM.
static int choose_tau(const std::vector<std::vector<int>>& cnt, int n, int S) {
    const double GEMM_RATE = 5.9e14;   // pair-bits / s  (campy, 128x256x512 tiles, L4)
    const double RARE_RATE = 3.5e9;    // rare pairs / s (atomicSub, L4)
    std::vector<int> c;
    for (auto& v : cnt) for (size_t a = 1; a < v.size(); a++) if (v[a]) c.push_back(v[a]);
    size_t freeb, totb; cudaMemGetInfo(&freeb, &totb);
    const size_t n_pad = up(n, 256), bufs = (size_t)S * n_pad * 12 + (64u << 20);
    int best = 0; double best_t = 1e300;
    for (int tau = 2; tau <= 4096; tau *= 2) {
        size_t D = 0; double R = 0;
        for (int x : c) { if (x >= tau) D++; else R += (double)x * (x - 1) / 2; }
        const size_t K1 = up(std::max<size_t>(D, 1), 1024);
        const double mem = (double)n_pad * K1 / 8 + 4.0 * (R + n + 1) + bufs;
        const double t = (double)n * n / 2 * K1 / GEMM_RATE + R / RARE_RATE;
        fprintf(stderr, "  tau=%5d  one-hot=%8zu  rare=%12.0f  est %.3f s  vram %.2f GB%s\n", tau, D, R, t, mem / 1e9,
                mem > freeb ? "  (does not fit)" : "");
        if (mem <= freeb && t < best_t) { best_t = t; best = tau; }
    }
    if (!best) { fprintf(stderr, "no tau fits in %.2f GB free VRAM\n", freeb / 1e9); exit(1); }
    fprintf(stderr, "auto tau = %d (est %.3f s)\n", best, best_t);
    return best;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <ranks> [tau] [stripe]\n", argv[0]); return 1; }
    int tau = argc > 2 ? atoi(argv[2]) : 0;  // 0 = auto (cost model)
    const int S = argc > 3 ? atoi(argv[3]) : 512;
    double t_all = now();

    FILE* f = fopen(argv[1], "rb");
    int nl[2];
    if (!f || fread(nl, 4, 2, f) != 2) { perror("read"); return 1; }
    const int n = nl[0], L = nl[1];
    std::vector<uint16_t> X((size_t)n * L);
    if (fread(X.data(), 2, X.size(), f) != X.size()) { perror("read"); return 1; }
    fclose(f);

    // ---- encode (OpenMP): dense one-hot columns for alleles seen >= tau times, rare groups otherwise ----
    double t_enc = now();
    std::vector<uint16_t> XT((size_t)L * n);  // locus-major copy for the per-locus passes
#pragma omp parallel for schedule(static)
    for (int i0 = 0; i0 < n; i0 += 256)
        for (int k = 0; k < L; k++)
            for (int i = i0; i < std::min(n, i0 + 256); i++) XT[(size_t)k * n + i] = X[(size_t)i * L + k];
    std::vector<std::vector<int>> col(L), cnt(L);  // locus k: rank -> dense column / count
    std::vector<size_t> base(L + 1, 0);
#pragma omp parallel for schedule(dynamic, 4)
    for (int k = 0; k < L; k++) {
        const uint16_t* x = &XT[(size_t)k * n];
        int mx = 0;
        for (int i = 0; i < n; i++) mx = std::max(mx, (int)x[i]);
        cnt[k].assign(mx + 1, 0);
        col[k].assign(mx + 1, -1);
        for (int i = 0; i < n; i++) cnt[k][x[i]]++;
    }
    if (tau <= 0) tau = choose_tau(cnt, n, S);
#pragma omp parallel for schedule(dynamic, 4)
    for (int k = 0; k < L; k++) {
        int c = 0;
        for (size_t a = 1; a < cnt[k].size(); a++) if (cnt[k][a] >= tau) col[k][a] = c++;
        base[k + 1] = c;
    }
    for (int k = 0; k < L; k++) base[k + 1] += base[k];
    const size_t D = base[L];
#pragma omp parallel for schedule(dynamic, 4)
    for (int k = 0; k < L; k++)
        for (auto& c : col[k]) if (c >= 0) c += (int)base[k];
    const size_t n_pad = up(n, 256), K1 = up(std::max<size_t>(D, 1), 1024), K2 = up(L, 1024);
    std::vector<uint8_t> O(n_pad * K1 / 8, 0), P(n_pad * K2 / 8, 0);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
        for (int k = 0; k < L; k++) {
            int a = X[(size_t)i * L + k];
            if (!a) continue;
            P[((size_t)i * K2 + k) >> 3] |= 1u << (k & 7);
            int c = col[k][a];
            if (c >= 0) O[((size_t)i * K1 + c) >> 3] |= 1u << (c & 7);
        }
    // rare equal pairs (i < j), bucketed by i: groups per locus, then count + fill with atomics
    std::vector<std::vector<std::vector<int>>> groups(L);
#pragma omp parallel for schedule(dynamic, 4)
    for (int k = 0; k < L; k++) {
        const uint16_t* x = &XT[(size_t)k * n];
        std::vector<std::vector<int>> g(cnt[k].size());
        for (int i = 0; i < n; i++) {
            int a = x[i];
            if (a && col[k][a] < 0 && cnt[k][a] >= 2) g[a].push_back(i);
        }
        for (auto& v : g) if (v.size() >= 2) groups[k].push_back(std::move(v));
    }
    std::vector<uint64_t> rp_cnt(n + 1, 0);
#pragma omp parallel for schedule(dynamic, 4)
    for (int k = 0; k < L; k++)
        for (auto& v : groups[k])
            for (size_t a = 0; a < v.size(); a++) __atomic_fetch_add(&rp_cnt[v[a]], v.size() - 1 - a, __ATOMIC_RELAXED);
    size_t n_rare = 0;
    std::vector<uint32_t> rp_start(n + 1, 0);
    for (int i = 0; i <= n; i++) { rp_start[i] = (uint32_t)n_rare; n_rare += rp_cnt[i]; }
    if (n_rare >= (1ull << 32)) { fprintf(stderr, "too many rare pairs (%zu): raise tau\n", n_rare); return 1; }
    std::vector<uint32_t> rp_j(n_rare);
    {
        std::vector<uint32_t> fill(rp_start.begin(), rp_start.end() - 1);
#pragma omp parallel for schedule(dynamic, 4)
        for (int k = 0; k < L; k++)
            for (auto& v : groups[k])
                for (size_t a = 0; a < v.size(); a++)
                    for (size_t b = a + 1; b < v.size(); b++)
                        rp_j[__atomic_fetch_add(&fill[v[a]], 1u, __ATOMIC_RELAXED)] = v[b];
    }
    groups.clear(); XT.clear(); XT.shrink_to_fit();
    t_enc = now() - t_enc;
    fprintf(stderr, "n=%d L=%d tau=%d: dense one-hot cols=%zu (K1=%zu bits), rare equal pairs=%zu, encode %.2fs\n",
            n, L, tau, D, K1, n_rare, t_enc);

    // ---- GPU: two streams, double-buffered stripes (GEMMs -> rare -> pack u16 -> async D2H) ----
    if (L > 65535) { fprintf(stderr, "L > 65535 not supported (u16 output)\n"); return 1; }
    const bool verify = !(argc > 4 && std::string(argv[4]) == "noverify");
    uint8_t *dO, *dP;
    const size_t ldd = n_pad;
    CK(cudaMalloc(&dO, O.size())); CK(cudaMalloc(&dP, P.size()));
    CK(cudaMemcpy(dO, O.data(), O.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dP, P.data(), P.size(), cudaMemcpyHostToDevice));
    // rare pairs on the GPU when they fit (leaving room for the stripe buffers), else on the host
    uint32_t *dStart = nullptr, *dJ = nullptr;
    size_t freeb, totb; CK(cudaMemGetInfo(&freeb, &totb));
    const bool rare_gpu = (n_rare + n + 1) * 4 + (size_t)S * ldd * 12 + (64u << 20) < freeb;
    if (rare_gpu) {
        CK(cudaMalloc(&dStart, (n + 1) * 4)); CK(cudaMalloc(&dJ, std::max<size_t>(n_rare, 1) * 4));
        CK(cudaMemcpy(dStart, rp_start.data(), (n + 1) * 4, cudaMemcpyHostToDevice));
        if (n_rare) CK(cudaMemcpy(dJ, rp_j.data(), n_rare * 4, cudaMemcpyHostToDevice));
    }
    int32_t* dH[2]; uint16_t* dU[2]; uint16_t* hU[2]; cudaStream_t st[2]; cudaEvent_t done[2];
    for (int b = 0; b < 2; b++) {
        CK(cudaMalloc(&dH[b], (size_t)S * ldd * 4)); CK(cudaMalloc(&dU[b], (size_t)S * ldd * 2));
        CK(cudaMallocHost(&hU[b], (size_t)S * ldd * 2));
        CK(cudaStreamCreateWithFlags(&st[b], cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&done[b], cudaEventDisableTiming));
    }
    fprintf(stderr, "rare pairs on %s, stripe %d rows, %s\n", rare_gpu ? "GPU" : "host", S,
            verify ? "verifying (full checksum + spot checks)" : "noverify (consumer does nothing)");

    gemm(256, 256, (int)K1, dO, dO, (int)K1, dH[0], dH[0], (int)ldd, 1, 0, st[0]);  // warm-up
    CK(cudaDeviceSynchronize());

    uint64_t h = 0;
    double t_host = 0;
    std::mt19937 rng(7);
    auto consume = [&](int r0) -> bool {
        const int rows = std::min(S, n - r0), b = (r0 / S) & 1;
        const size_t w = n - r0;
        uint16_t* H = hU[b];
        double th = now();
        if (!rare_gpu)
            for (int r = 0; r < rows; r++) {
                int i = r0 + r;
                for (uint32_t p = rp_start[i]; p < rp_start[i + 1]; p++) H[(size_t)r * w + (rp_j[p] - r0)] -= 1;
            }
        if (verify) {
            for (int r = 0; r < rows; r++)
                for (size_t j = r + 1; j < w; j++) h = h * 1000003u + H[(size_t)r * w + j];
            for (int t = 0; t < 4; t++) {  // spot-check against a direct count
                int r = rng() % rows; size_t j = r + rng() % (w - r);
                int i = r0 + r, jj = r0 + (int)j, d = 0;
                for (int k = 0; k < L; k++) {
                    int a = X[(size_t)i * L + k], c = X[(size_t)jj * L + k];
                    d += (a && c && a != c);
                }
                if (d != H[(size_t)r * w + j]) {
                    fprintf(stderr, "MISMATCH i=%d j=%d tc=%d ref=%d\n", i, jj, H[(size_t)r * w + j], d);
                    return false;
                }
            }
        }
        t_host += now() - th;
        return true;
    };
    double t_pipe = now();
    int prev = -1;
    for (int r0 = 0; r0 < n; r0 += S) {
        const int rows = std::min(S, n - r0), b = (r0 / S) & 1;
        const int M = (int)up(rows, 8), N = (int)up(n - r0, 8);
        const size_t w = n - r0;
        gemm(M, N, (int)K1, dO + (size_t)r0 * K1 / 8, dO + (size_t)r0 * K1 / 8, (int)K1, dH[b], dH[b], (int)ldd, 1, 0, st[b]);
        gemm(M, N, (int)K2, dP + (size_t)r0 * K2 / 8, dP + (size_t)r0 * K2 / 8, (int)K2, dH[b], dH[b], (int)ldd, 1, -1, st[b]);
        if (rare_gpu) sub_rare<<<(rows + 7) / 8, 256, 0, st[b]>>>(dH[b], ldd, r0, rows, dStart, dJ);
        pack_u16<<<1024, 256, 0, st[b]>>>(dH[b], ldd, dU[b], w, rows);
        CK(cudaMemcpyAsync(hU[b], dU[b], (size_t)rows * w * 2, cudaMemcpyDeviceToHost, st[b]));
        CK(cudaEventRecord(done[b], st[b]));
        if (prev >= 0) {  // consume the previous stripe while this one runs
            CK(cudaEventSynchronize(done[(prev / S) & 1]));
            if (!consume(prev)) return 2;
        }
        prev = r0;
    }
    CK(cudaEventSynchronize(done[(prev / S) & 1]));
    if (!consume(prev)) return 2;
    t_pipe = now() - t_pipe;
    printf("tc_hamming tile=%dx%dx%d tau=%d: pipeline (GPU + D2H u16 + consumer) %.3f s, of which consumer %.2f s, "
           "encode %.2f s, total %.2f s  checksum=%llx\n",
           TB_M, TB_N, TB_K, tau, t_pipe, t_host, t_enc, now() - t_all, (unsigned long long)h);
    return 0;
}
