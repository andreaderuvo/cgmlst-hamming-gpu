# tc-hamming — exact cgMLST Hamming distances on 1-bit tensor cores

`tc-hamming` computes the full pairwise **Hamming distance matrix between
cgMLST allelic profiles** — for every pair of isolates, the number of loci where
both alleles are called and differ — on the GPU's **1-bit tensor cores**, the
AND+popcount units built for binarized neural networks. No training, no
approximation: the result is exact, integer for integer.

On 63 005 *Campylobacter* profiles × 1 142 loci (≈ 2 billion pairs) the whole
run, TSV in → full TSV matrix out, takes **5.6 s** on an NVIDIA L4, of which
**0.7 s** is the distance computation.

Project page: <https://andreaderuvo.github.io/tc-hamming/>

## How it works

For two profiles *a*, *b* and a locus *k* (0 = missing):

```
[a≠0][b≠0][a≠b] = [a≠0][b≠0] − [a=b≠0]
```

so, summed over loci, the distance matrix is

```
H = C − E
C = loci called in both            = P Pᵀ        P: n × L presence bits
E = loci called in both and equal  = O Oᵀ + R    O: one-hot bits of alleles seen ≥ τ times
                                                 R: equal pairs of rarer alleles (sparse, enumerated)
```

- **P Pᵀ and O Oᵀ are binary matrix products** — exactly what 1-bit tensor
  cores do (AND, popcount, int32 accumulation). The second product's epilogue
  computes `P Pᵀ − O Oᵀ` in place.
- **One-hot only for frequent alleles.** Equality between many categories
  needs a one-hot expansion; real cgMLST data have a long tail of rare alleles
  that would blow it up. Alleles seen fewer than τ times are left out of the
  one-hot and their equal pairs are enumerated exactly and subtracted
  (`atomicSub` on the GPU).
- **τ is a cost knob, not an accuracy knob**: it only moves work between the
  GEMM and the sparse term. It is chosen automatically from a cost model (GEMM
  bit-throughput vs rare-pair rate, constrained to the free VRAM).
- Only the upper triangle is computed, in row stripes, on two CUDA streams:
  while the host formats stripe *s*, stripe *s+1* is computed and copied back.

### Why it is always exact

1. The identity above holds for every pair of integers.
2. For a locus, `Σ_v [a=v][b=v] = [a=b]`, so summing the one-hot products over
   all alleles counts equal called pairs; frequent and rare alleles are a
   disjoint partition of the alleles, so `O Oᵀ + R` counts each equal pair
   exactly once, for any τ.
3. All arithmetic is integer (bits, popcounts, int32 accumulators; distances
   ≤ L < 65 536 stored as u16); zero padding adds nothing.
4. Allele ids are mapped to dense per-locus ranks by an injective map that
   keeps 0 = missing, so equality and missingness are preserved (hashed
   32/64-bit ids are fine).

`tensorcore/rs/check.py` checks every cell against a brute-force count on
inputs with missing, empty, `INF-`, non-numeric and hashed calls, for several
τ, stripe sizes and output layouts.

## Usage

Requirements: NVIDIA GPU with tensor cores (sm_80+; developed on sm_89), CUDA
toolkit, Rust, and CUTLASS headers.

```bash
git clone https://github.com/andreaderuvo/tc-hamming && cd tc-hamming
git clone --depth 1 https://github.com/NVIDIA/cutlass.git tensorcore/third_party/cutlass
cd tensorcore/rs
cargo build --release            # CUDA_ARCH=sm_80 / CUTLASS_DIR=... to override
./target/release/tc-hamming -i calls.tsv -o dists.tsv
python3 check.py                 # exactness check
```

Input: allele-call TSV (chewBBACA style), first column sample id, one column
per locus; `-`, empty and non-numeric calls are missing, `INF-` is stripped.

| option | meaning |
|---|---|
| `-i`, `-o` | input (`-` = stdin), output (default stdout; `.gz` → gzip) |
| `-m full\|lower-tri\|upper-tri` | matrix layout |
| `-d`, `-D` | input / output separator |
| `-n` | missing-data character (default `-`) |
| `-r` | do not strip `INF-` |
| `-L`, `-S` | keep loci / samples with at least this % of calls |
| `-X` | cap distances at this value |
| `-x` | top-left cell label |
| `-z` | gzip output |
| `--tau` | frequency threshold (0 = automatic) |
| `--stripe`, `--device`, `-t`, `-s` | rows per GPU stripe, CUDA device, CPU threads, silent |

## Results

NVIDIA L4 (shared with other jobs, ~1.3–1.8 GB of VRAM free), 80-core CPU.

**End to end**, *Campylobacter*, 63 005 × 1 142 real chewBBACA calls:

| step | time |
|---|---:|
| parse TSV | 0.35 s |
| encode (ranks, one-hot, rare pairs) | 0.33 s |
| tensor-core distances + copy back | 0.7 s |
| write the 63k × 63k TSV (~8 GB) | 3.9 s |
| **total** | **5.6 s** |

On a 40 000-profile subset the whole run takes 2.6 s. Writing text is now the
bottleneck.

**Tensor cores vs a direct-compare GPU kernel** (same GPU, tiled CUDA kernel
comparing alleles locus by locus, `cuda/hamming.cu`); time = GPU compute + u16
copy back:

| dataset | direct compare | tensor cores | identical |
|---|---:|---:|:---:|
| *Campylobacter* 63 005 × 1 142 (real) | 1.79 s | **0.365 s** (τ = 64, auto) | ✓ |
| synthetic 30 000 × 5 000 | 1.74 s | **0.202 s** (τ = 8, auto) | ✓ |
| synthetic 50 000 × 5 000 | 4.85 s | does not fit in the free VRAM | — |

Measured peaks on this L4: int8 GEMM 200 TOPS, 1-bit AND+popcount 1 924 TOPS.

**Limit**: the frequent-allele one-hot must fit in VRAM (no K-tiling yet). Real
data with a long tail of rare alleles fit easily; the synthetic 50k (50
founder alleles per locus, all frequent: 221 k one-hot columns ≈ 1.4 GB) does
not on the shared GPU, but would on a free 24 GB card.

## What is and isn't new

The pieces are known: one-hot × one-hot = match count; the frequent/rare split
is the heavy/light technique for join-project queries (Amossen & Pagh, ICDT
2009; Deep, Hu & Koutris, SIGMOD 2020); binary tensor cores have been used in
genomics for epistasis on biallelic SNPs (Nobre et al.) and XOR+popcount
Hamming on bit-packed genotypes is classic (PLINK 1.9). What this project adds
is the combination: exact multi-allelic Hamming with missing data on 1-bit
tensor cores, with the frequent/rare split as the cost knob. Details and
references: [`tensorcore/README.md`](tensorcore/README.md).

## Repository

| path | what |
|---|---|
| `tensorcore/rs/` | **tc-hamming**: Rust host (parsing, filters, encoding, auto-τ, parallel writer) + CUDA/CUTLASS kernel (`cuda/tch.cu`) via FFI |
| `tensorcore/` | C++ prototype (`tc_hamming.cu`), tensor-core peak benchmark, allele-frequency cost model, notes |
| `cuda/`, `python/`, [`RELU.md`](RELU.md) | where it started: Hamming as a hand-wired ReLU network (exact, but 2.3× slower than a plain compare — abandoned) |
| `docs/` | project page |

## License

GPL-3.0 — see [`LICENSE`](LICENSE).
