# Side exploration: Hamming distance as a hand-wired ReLU network

Research sandbox: the cgMLST Hamming distance (count of loci where both
alleles are called and differ) written as an **exact** ReLU network — weights
set by hand, no training — then run at scale on a GPU and compared with a
plain comparison.

## The construction

Per locus, with `a`, `b` integer allele ids (0 = missing):

```
diff = relu(a-b) + relu(b-a) - relu(a-b-1) - relu(b-a-1)   = min(1, |a-b|) = [a != b]
pa   = a - relu(a-1)                                         = min(1, a)     = [a > 0]
pb   = b - relu(b-1)                                         = min(1, b)     = [b > 0]
out  = relu(diff + pa + pb - 2)                              = AND of the three bits
distance = sum of out over loci
```

Three layers, weights in {-1, 0, 1}, integer biases. It is exact **only on
integers**: on the reals `[a != b]` is discontinuous and no ReLU network
computes it (between two integers the network is a ramp). Allele ids are
re-encoded as dense per-locus ranks first (equality is all the distance
needs), so hashed CRC32 ids stay exact in fp32.

## Contents

| path | what |
|---|---|
| `python/relu_hamming.py` | numpy reference: the network as weight matrices, checked against `!=` on random matrices and edge cases |
| `python/relu_gpu.py` | first GPU attempt, `torch.compile` (untiled, slow) |
| `python/relu_triton.py` | Triton kernel tiled like a GEMM, network in place of multiply-add |
| `python/relu_stripes.py` | 50k-scale Triton run in row stripes (fits beside other jobs on the GPU) |
| `python/gen_profiles.py` | synthetic clonal-ish cgMLST profiles (`gen_profiles.py <n> <L> <out.tsv>`) |
| `cuda/hamming.cu` | CUDA kernel, templated on the per-locus op (ReLU network or compare); stripes sized to free VRAM, double-buffered copy-back |
| `cuda/bench.cu` | benchmark of the two ops on the same kernel + identity check |

```bash
python3 python/relu_hamming.py          # exactness check, CPU
make && ./bench 50000 5000              # ReLU vs compare on the GPU
```

## Results (NVIDIA L4 shared with other jobs: < 700 MB VRAM used)

| 50 000 × 5 000 | time |
|---|---:|
| Triton, ReLU network | 18.1 s |
| CUDA, ReLU network | 11.6 s |
| CUDA, direct compare (same kernel) | **5.0 s** |

All outputs exact (the Triton output was checked byte-identical against a CPU
direct-compare reference at 5k and 50k; the two CUDA ops give the same matrix). **The speed comes from the GPU and the
tiling, not from ReLU**: the network costs ~11 ops per locus pair vs ~3 for a
comparison, hence 2.3× slower. The point of the ReLU form is expressivity, not
speed.

## Prior art (searched 2026-11-02)

- M. Ohtomo, A. Takasu, T. Akutsu, *Computing Hamming Distance and Levenshtein
  Distance Using ReLU Neural Networks*, IEEE Access 13:210089–210101 (2025),
  doi:10.1109/ACCESS.2025.3639297 — exact ReLU networks, O(n) units for
  Hamming, O(mn) for Levenshtein. **ReLU-Hamming itself is not new**; full text
  not yet read (encoding details unverified).
- M. Ghafoor, T. Akutsu — exact ReLU generators for trees within a tree edit
  distance (arXiv 2510.10706, 2025) and for graphs (arXiv 2604.05929, 2026).
- Compiling programs into weights: RASP (Weiss et al. 2021), Tracr (Lindner et
  al. 2023), ALTA (TMLR 2025), *Weights to Code* (arXiv 2601.05770, 2026),
  torchwright, Turing-complete PyTorch library (arXiv 2605.08150).
- Learned approximations instead of exact constructions: NeuroSEED (Corso et
  al. 2021).

Not found: exact ReLU form of a **Hamming-1NN classifier** (min is ReLU too:
`min(x,y) = x - relu(x-y)`), i.e. a capacity argument that ML classifiers on cgMLST
profiles contain Hamming-NN as a special case.

## Open directions

1. Hamming-1NN (distances → min → vote) as an exact ReLU network, checked on
   real data.
2. Learnability: train an MLP on profile pairs — does it recover these weights,
   and does it generalise to unseen allele ids?
3. Formal verification (MILP/SMT, e.g. Marabou) that the network equals Hamming
   on the whole input range.

## Literature check 2 — where else could exact ReLU matter? (2026-11-02)

Key fact: `max(x,y) = y + relu(x-y)` is exact on the **reals**, so every
max-plus / min-plus DP (Smith-Waterman, Needleman-Wunsch, Viterbi, Zuker/Nussinov
folding) is an exact ReLU network (ReLU nets = tropical rational functions:
Zhang, Naitzat, Lim, ICML 2018). No paper found that compiles SW/NW into a
ReLU net explicitly — but it is a direct corollary, and the payoff is taken:

- **Learning DP parameters through the exact (hard-max) algorithm is ~20 years
  old**: inverse parametric alignment (Kececioglu & Kim 2006–2008), structured
  SVM / max-margin (CONTRAfold, MXfold2 — Sato et al., Nat. Commun. 2021). The
  subgradient of a compiled ReLU DP is exactly that structured-perceptron/SSVM
  update, so the ReLU framing adds nothing there.
- Smoothed DP (logsumexp): SMURF / LAM, Petti et al., Bioinformatics 2023;
  differentiable folding of ~13k Turner parameters, Krueger, Aviran, Mathews,
  Zuber, Ward, arXiv 2503.09085 (2025).
- Tropical Attention (Hashemi, Pasque, Teska, Yoshida, NeurIPS 2025): *learned*
  max-plus attention for combinatorial/DP reasoning, not exact compilation.
- DNA LMs break exact symmetries (reverse complement): fixed by *regularisation*
  (RCCR, arXiv 2509.18529), not by compiled exact modules — the least occupied
  angle found, but speculative.
- CRISPR guide design with exact mismatch constraints via gradients: nothing
  found (weak search, not conclusive).

Verdict: for exact distances (our need) ReLU gives no advantage; for alignment /
folding parameter learning the field already does it without ReLU.
