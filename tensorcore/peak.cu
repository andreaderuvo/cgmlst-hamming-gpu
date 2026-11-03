// Peak tensor-core throughput on this GPU: int8 GEMM via cuBLAS, and 1-bit
// AND+popcount mma (PTX m16n8k256) in a register-only loop.
#include <cstdio>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime.h>

__global__ void b1_peak(int iters, int* out) {
    uint32_t a0 = threadIdx.x * 2654435761u, a1 = a0 ^ 0x9e3779b9u, a2 = a0 * 3u, a3 = a0 + 7u;
    uint32_t b0 = a1 * 5u, b1 = a2 ^ 0x55555555u;
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int u = 0; u < 8; u++)
            asm volatile("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                         : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    if (c0 + c1 + c2 + c3 == 42) out[0] = 1;
}

int main() {
    int dev; cudaGetDevice(&dev); cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    printf("%s sm_%d%d, %d SMs\n", p.name, p.major, p.minor, p.multiProcessorCount);
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1); float ms;

    // ---- 1-bit mma ----
    int* out; cudaMalloc(&out, 4);
    int blocks = p.multiProcessorCount * 8, threads = 128, iters = 20000;
    b1_peak<<<blocks, threads>>>(100, out); cudaDeviceSynchronize();
    cudaEventRecord(t0); b1_peak<<<blocks, threads>>>(iters, out); cudaEventRecord(t1);
    cudaEventSynchronize(t1); cudaEventElapsedTime(&ms, t0, t1);
    if (cudaGetLastError() != cudaSuccess) { printf("b1 mma failed\n"); }
    double warps = blocks * threads / 32.0;
    double ops = warps * iters * 8 * (16.0 * 8 * 256) * 2;  // AND + popc-add per bit
    printf("b1 AND.popc mma : %8.1f TOPS (bit-ops)\n", ops / ms / 1e9);

    // ---- int8 GEMM via cuBLAS ----
    cublasHandle_t h; cublasCreate(&h);
    int M = 8192, N = 8192, K = 8192; int8_t *A, *B; int32_t* C;
    cudaMalloc(&A, (size_t)M * K); cudaMalloc(&B, (size_t)N * K); cudaMalloc(&C, (size_t)M * N * 4);
    cudaMemset(A, 1, (size_t)M * K); cudaMemset(B, 1, (size_t)N * K);
    int32_t alpha = 1, beta = 0;
    auto run = [&]() {
        return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, A, CUDA_R_8I, K, B, CUDA_R_8I, K,
                            &beta, C, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    };
    if (run() != CUBLAS_STATUS_SUCCESS) { printf("int8 gemm failed\n"); return 1; }
    cudaDeviceSynchronize();
    cudaEventRecord(t0); for (int r = 0; r < 10; r++) run(); cudaEventRecord(t1);
    cudaEventSynchronize(t1); cudaEventElapsedTime(&ms, t0, t1);
    int32_t c00; cudaMemcpy(&c00, C, 4, cudaMemcpyDeviceToHost);
    printf("int8 GEMM cuBLAS: %8.1f TOPS  (check C[0]=%d, expect %d)\n", 10 * 2.0 * M * N * K / ms / 1e9, c00, K);
}
