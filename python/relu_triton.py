import sys, time, numpy as np, pandas as pd, torch, triton, triton.language as tl

@triton.jit
def relu_hamming_kernel(A, B, D, n, L, sa, sd,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pi, pj = tl.program_id(0), tl.program_id(1)
    if pj < pi:                                   # symmetric: lower tiles skipped
        return
    rm = pi * BM + tl.arange(0, BM); rn = pj * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, L, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(A + rm[:, None] * sa + rk[None, :], mask=(rm[:, None] < n) & (rk[None, :] < L), other=0.)
        b = tl.load(B + rn[:, None] * sa + rk[None, :], mask=(rn[:, None] < n) & (rk[None, :] < L), other=0.)
        x = a[:, None, :]; y = b[None, :, :]                      # (BM,1,BK), (1,BN,BK)
        # ---- the hand-wired ReLU network, per locus ----
        d = x - y
        diff = tl.maximum(d, 0.) + tl.maximum(-d, 0.) - tl.maximum(d - 1., 0.) - tl.maximum(-d - 1., 0.)
        pa = x - tl.maximum(x - 1., 0.)
        pb = y - tl.maximum(y - 1., 0.)
        acc += tl.sum(tl.maximum(diff + pa + pb - 2., 0.), axis=2)
    out = acc.to(tl.int32)
    msk = (rm[:, None] < n) & (rn[None, :] < n)
    tl.store(D + rm[:, None] * sd + rn[None, :], out, mask=msk)
    tl.store(D + rn[None, :] * sd + rm[:, None], out, mask=msk)  # mirror

def load(inp):
    df = pd.read_csv(inp, sep="\t", index_col=0, dtype=str)
    codes = np.zeros(df.shape, dtype=np.float32)
    for j, c in enumerate(df.columns):
        num = pd.to_numeric(df[c].str.replace("INF-", "", regex=False), errors="coerce")
        num[num == 0] = np.nan
        codes[:, j] = pd.factorize(num)[0] + 1
    return df.index.tolist(), codes

def run(X, BM=32, BN=32, BK=8, warps=4):
    n, L = X.shape
    D = torch.empty((n, n), dtype=torch.int32, device=X.device)
    g = (triton.cdiv(n, BM), triton.cdiv(n, BN))
    relu_hamming_kernel[g](X, X, D, n, L, X.stride(0), D.stride(0), BM=BM, BN=BN, BK=BK, num_warps=warps)
    return D

if __name__ == "__main__":
    inp, out = sys.argv[1], sys.argv[2]
    t = time.time(); names, codes = load(inp); tp = time.time() - t
    X = torch.from_numpy(codes).cuda()
    cfg = dict(BM=int(sys.argv[3]), BN=int(sys.argv[3]), BK=int(sys.argv[4]), warps=int(sys.argv[5])) if len(sys.argv) > 3 else {}
    run(X[:64], **cfg); torch.cuda.synchronize()
    t = time.time(); D = run(X, **cfg); torch.cuda.synchronize(); tg = time.time() - t
    t = time.time(); Dn = D.cpu().numpy()
    with open(out, "w") as f:
        f.write("cgmlst-hamming-gpu\t" + "\t".join(names) + "\n")
        for nm, row in zip(names, Dn): f.write(nm + "\t" + "\t".join(map(str, row.tolist())) + "\n")
    tw = time.time() - t
    n, L = codes.shape
    print(f"n={n} L={L} parse={tp:.2f}s gpu={tg:.3f}s ({n*n/2*L/tg/1e9:.0f} G locus-pairs/s) write={tw:.2f}s", file=sys.stderr)
