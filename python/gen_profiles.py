import sys, numpy as np
n, L, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
rng = np.random.default_rng(1)
# clonal-ish: 50 founders, each sample mutates ~3% of loci, 2% missing
F = rng.integers(1, 200, size=(50, L))
P = F[rng.integers(0, 50, n)]
m = rng.random((n, L)) < 0.03; P[m] = rng.integers(1, 5000, m.sum())
P[rng.random((n, L)) < 0.02] = 0
with open(out, "w") as f:
    f.write("FILE\t" + "\t".join(f"L{j}" for j in range(L)) + "\n")
    for i, r in enumerate(P):
        f.write(f"S{i}\t" + "\t".join("-" if v == 0 else str(v) for v in r) + "\n")
