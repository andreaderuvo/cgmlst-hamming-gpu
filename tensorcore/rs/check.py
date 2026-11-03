#!/usr/bin/env python3
"""Exactness check of cgmlst-hamming-gpu against a brute-force numpy reference.

Builds allele-call TSVs with the awkward cases (missing '-', empty cells,
`INF-` prefixes, non-numeric calls, 32-bit hashed ids, very frequent and
singleton alleles), runs cgmlst-hamming-gpu with several τ / stripe / layout options
and compares every cell with a direct pairwise count.

    python3 check.py [path/to/cgmlst-hamming-gpu]
"""
import os, subprocess, sys, tempfile
import numpy as np

TC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "target/release/cgmlst-hamming-gpu")


def make(n, L, seed, hashed):
    rng = np.random.default_rng(seed)
    founders = rng.integers(1, 30, size=(8, L))
    A = founders[rng.integers(0, 8, n)]
    m = rng.random((n, L)) < 0.15
    A[m] = rng.integers(1, 2000, m.sum())  # long tail of rare alleles
    A[rng.random((n, L)) < 0.05] = 0
    if hashed:  # CRC32-like ids: equality is all that matters
        A = np.where(A > 0, (A * 2654435761) % 4294967291 + 1, 0)
    cells = []
    for r in A:
        row = []
        for v in r:
            u = rng.random()
            if v == 0:
                row.append("-" if u < 0.4 else "" if u < 0.6 else "LNF" if u < 0.8 else "PLOT3")
            else:
                row.append(f"INF-{v}" if u < 0.1 else str(v))
        cells.append(row)
    return A, cells


def ref(A):
    P = A > 0
    H = np.zeros((len(A), len(A)), dtype=np.int64)
    for i in range(len(A)):
        H[i] = ((A[i] != A) & P[i] & P).sum(1)
    return H


def run(path, args):
    out = subprocess.run([TC, "-s", "-i", path, *args], capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(f"cgmlst-hamming-gpu {args} failed: {out.stderr}")
    lines = out.stdout.rstrip("\n").split("\n")
    sep = "," if "-D" in args else "\t"
    return np.array([[int(x) for x in l.split(sep)[1:]] for l in lines[1:]])


def main():
    fails = 0
    cases = [(300, 97, 1, False), (700, 410, 2, True), (1500, 64, 3, False)]
    with tempfile.TemporaryDirectory() as d:
        for n, L, seed, hashed in cases:
            A, cells = make(n, L, seed, hashed)
            path = os.path.join(d, f"c{seed}.tsv")
            with open(path, "w") as f:
                f.write("FILE\t" + "\t".join(f"L{j}" for j in range(L)) + "\n")
                for i, row in enumerate(cells):
                    f.write(f"S{i}\t" + "\t".join(row) + "\n")
            H = ref(A)
            iu = np.triu(np.ones_like(H, dtype=bool))
            for args, want in [
                ([], H),
                (["--tau", "2"], H),
                (["--tau", "100000"], H),
                (["--stripe", "64"], H),
                (["-m", "lower-tri"], np.where(iu.T, H, 0)),
                (["-m", "upper-tri"], np.where(iu, H, 0)),
                (["-X", "7"], np.minimum(H, 7)),
                (["-D", ","], H),
            ]:
                ok = np.array_equal(run(path, args), want)
                fails += not ok
                print(f"{'OK  ' if ok else 'FAIL'} n={n} L={L} hashed={hashed} {' '.join(args) or '(default)'}")
    print("all exact" if not fails else f"{fails} FAILED")
    sys.exit(bool(fails))


if __name__ == "__main__":
    main()
