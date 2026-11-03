# cgMLST TSV -> dense per-locus ranks (u16, 0 = missing), row-major, with a small header
import sys, numpy as np, pandas as pd
df = pd.read_csv(sys.argv[1], sep="\t", index_col=0, dtype=str)
X = np.zeros(df.shape, dtype=np.uint16)
for j, c in enumerate(df.columns):
    v = pd.to_numeric(df[c].str.replace("INF-", "", regex=False), errors="coerce")
    v[v <= 0] = np.nan
    X[:, j] = pd.factorize(v)[0] + 1
with open(sys.argv[2], "wb") as f:
    np.array(X.shape, dtype=np.int32).tofile(f); X.tofile(f)
print(X.shape, X.max())
