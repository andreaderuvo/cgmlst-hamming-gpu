# One-hot cost model for exact Hamming = C - E on real cgMLST data.
# Alleles seen >= tau times -> dense one-hot columns (tensor-core GEMM);
# rarer alleles -> their equal pairs enumerated sparsely.
import sys, numpy as np, pandas as pd
df = pd.read_csv(sys.argv[1], sep="\t", index_col=0, dtype=str)
n, L = df.shape
cnts = []
miss = 0
for c in df.columns:
    v = pd.to_numeric(df[c].str.replace("INF-", "", regex=False), errors="coerce")
    v = v[v > 0]
    miss += n - len(v)
    cnts.append(v.value_counts().values)
allc = np.concatenate(cnts)
print(f"n={n} L={L} missing={miss/(n*L):.2%} distinct alleles total={len(allc)} "
      f"median/locus={np.median([len(c) for c in cnts]):.0f} max/locus={max(len(c) for c in cnts)}")
for tau in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    dense = allc[allc >= tau]; rare = allc[allc < tau]
    rare_pairs = int((rare * (rare - 1) // 2).sum())
    cover = dense.sum() / allc.sum()
    print(f"tau={tau:4d}  dense cols={len(dense):8d} ({len(dense)/L:6.1f}/locus)  "
          f"calls covered={cover:6.2%}  rare equal-pairs={rare_pairs:,}")
