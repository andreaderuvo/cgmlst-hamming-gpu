# tensorcore — exact cgMLST Hamming distance on 1-bit tensor cores

Exact pairwise Hamming distance between cgMLST allelic profiles (loci where
both alleles are called and differ; 0 = missing) computed with **1-bit
AND+popcount tensor-core GEMMs** — the deep-learning hardware, no training.

## Method

```
H(i,j) = C(i,j) - E(i,j)
C = loci called in both            = P Pᵀ          P: n × L presence bits
E = loci called in both and equal  = O Oᵀ + R      O: one-hot bits of alleles seen ≥ τ times
                                                   R: equal pairs of rarer alleles (sparse)
```

Equality between *a* categories needs rank *a* as a bilinear form, so a one-hot
expansion cannot be avoided; it is made affordable by (1) one-hot only for
alleles seen ≥ τ times, the rest handled as an exact sparse correction, and
(2) bit-packed operands on the 1-bit tensor cores (~10× int8 throughput on an
L4). Both GEMMs accumulate in int32 (exact); the second GEMM's epilogue
computes `P Pᵀ − O Oᵀ` in place. Upper triangle only, in row stripes sized to
the free VRAM.

Files: `tc_hamming.cu` (pipeline, CUTLASS), `peak.cu` (tensor-core peak
microbenchmark), `allele_stats.py` (one-hot cost model per τ), `to_ranks.py`
(TSV → dense per-locus u16 ranks), `bench_file.cu` / `bench_noop.cu`
(direct-compare reference kernel from `../cuda/hamming.cu`).

```bash
git clone --depth 1 https://github.com/NVIDIA/cutlass.git third_party/cutlass
nvcc -O3 -std=c++17 -arch=sm_89 -Xcompiler -fopenmp -Ithird_party/cutlass/include -Ithird_party/cutlass/tools/util/include \
     -DTB_M=128 -DTB_N=256 -DTB_K=512 -DW_M=64 -DW_N=64 -DSTAGES=3 tc_hamming.cu -o tc_hamming -lgomp
python3 to_ranks.py calls.tsv calls.ranks
./tc_hamming calls.ranks            # tau auto, stripe 512, full checksum + spot checks
./tc_hamming calls.ranks 0 512 noverify   # timing only
```

## The tool

`rs/` is the usable tool (tc-hamming): Rust host (parsing, filters, encoding,
auto-τ, parallel TSV/gzip writer) + the CUDA/CUTLASS kernel via FFI. See the
top-level README for usage and end-to-end timings; `rs/check.py` checks it
cell by cell against a brute-force reference.

## Results of the prototype (NVIDIA L4 shared with other jobs: ~1.3–1.8 GB VRAM free)

Peaks measured on this L4: int8 GEMM (cuBLAS) 200 TOPS, 1-bit AND.popc mma 1924 TOPS.

| dataset | direct-compare GPU kernel: GPU + D2H u16 | **tensor cores: GPU + D2H u16** | tensor cores: whole run (read ranks, encode, GPU) | exact |
|---|---:|---:|---:|:---:|
| Campylobacter 63 005 × 1 142 (real) | 1.79 s | **0.365 s** (τ = 64 auto) | 1.2 s | ✓ `f8c5061206c302d0` |
| synthetic 30 000 × 5 000 | 1.74 s | **0.202 s** (τ = 8 auto) | 1.1 s | ✓ `85724a9d7d090f9f` |
| synthetic 50 000 × 5 000 | 4.85 s | does not fit: 221 k frequent one-hot columns ≈ 1.4 GB > 1.3 GB free | — | — |

Exact = upper-triangle checksum identical to the direct-compare kernel, plus
per-stripe spot checks against a direct count. Machine load varied during the
runs (load average up to ~40).

Limit: the frequent-allele one-hot must fit in VRAM (no K-tiling yet). On the
shared L4 the synthetic 50k (50 founder alleles per locus, all frequent) does
not; on a free 24 GB L4 it would. Real data with a long tail of rare alleles
(campy) fits comfortably.

## Prior art (searched 2026-11-02)

- **Binary tensor cores in genomics exist for epistasis detection**: Nobre et
  al., *Exploring the Binary Precision Capabilities of Tensor Cores for
  Epistasis Detection* (and follow-ups, e.g. Marques et al., *Unlocking
  Personalized Healthcare on Modern CPUs/GPUs: Three-way Gene Interaction
  Study*, arXiv 2201.10956). There the data are biallelic SNP genotypes,
  already binary — no categorical expansion, no missing-data semantics.
- **Distances on tensor cores**: FASTED (arXiv 2508.21230), mixed-precision
  Euclidean distances. Approximate floating point, not exact categorical
  mismatch counts.
- One-hot × one-hot = match count is textbook; nothing found applying it
  exactly to multi-allelic profiles with missing data on tensor cores, nor for
  cgMLST.

- **The frequency split is the classic heavy/light trick of database theory**:
  E = O Oᵀ is a join-project count over (isolate, locus, allele); splitting
  values into heavy (dense matrix multiplication) and light (enumerated) is
  Amossen & Pagh, *Faster join-projects and sparse matrix multiplications*,
  ICDT 2009, and Deep, Hu, Koutris, *Fast Join Project Query Evaluation using
  Matrix Multiplication*, SIGMOD 2020. **The algorithmic core is not new.**
- XOR+popcount Hamming on bit-packed biallelic genotypes: PLINK 1.9 (Chang et
  al. 2015); bit tensor cores for binarized neural networks (Li et al., 2020).

What is new here is only the combination: exact multi-allelic Hamming with missing data on 1-bit
tensor cores via a frequency split (dense one-hot for frequent alleles, exact
sparse correction for rare ones) with τ as the cost knob.
