import sys, time, numpy as np, pandas as pd, torch
torch.set_float32_matmul_precision("high")
inp, out = sys.argv[1], sys.argv[2]
T0 = time.time()
df = pd.read_csv(inp, sep="\t", index_col=0, dtype=str)
names = df.index.tolist()
# per-locus dense ranks: missing -> 0, alleles -> 1..k  (equality preserved, exact in fp32)
codes = np.zeros(df.shape, dtype=np.float32)
for j, c in enumerate(df.columns):
    col = df[c].str.replace("INF-", "", regex=False)
    num = pd.to_numeric(col, errors="coerce")             # LNF, PLOT, '-' -> NaN -> missing
    num[num == 0] = np.nan
    codes[:, j] = pd.factorize(num)[0] + 1                 # NaN -> -1 -> 0
T_parse = time.time() - T0

relu = torch.relu
def locus_net(a, b):                       # the hand-wired ReLU network, per locus
    h0, h1 = relu(a - b), relu(b - a)
    h2, h3 = relu(a - b - 1), relu(b - a - 1)
    diff = h0 + h1 - h2 - h3               # [a != b]
    pa, pb = a - relu(a - 1), b - relu(b - 1)   # [a > 0], [b > 0]
    return relu(diff + pa + pb - 2)        # AND
@torch.compile(dynamic=False)
def block(A, B):                           # (bi,L),(bj,L) -> (bi,bj), fused reduction
    return locus_net(A[:, None, :], B[None, :, :]).sum(-1)

dev = "cuda"
X = torch.from_numpy(codes).to(dev)
n, L = X.shape
bs = int(sys.argv[3]) if len(sys.argv) > 3 else 512
D = torch.empty((n, n), dtype=torch.int32, device="cpu", pin_memory=True)
pad = (-n) % bs
Xp = torch.cat([X, torch.zeros(pad, L, device=dev)]) if pad else X
block(Xp[:bs], Xp[:bs]); torch.cuda.synchronize()        # compile warm-up
T1 = time.time()
for i in range(0, n, bs):
    for j in range(i, n, bs):                              # symmetric: upper blocks only
        r = block(Xp[i:i+bs], Xp[j:j+bs]).to(torch.int32)
        ri, rj = min(bs, n-i), min(bs, n-j)
        D[i:i+ri, j:j+rj] = r[:ri, :rj].cpu()
        if j != i: D[j:j+rj, i:i+ri] = r[:ri, :rj].T.cpu()
torch.cuda.synchronize(); T_gpu = time.time() - T1
T2 = time.time()
Dn = D.numpy()
with open(out, "w") as f:
    f.write("cgmlst-hamming-gpu\t" + "\t".join(names) + "\n")
    for nm, row in zip(names, Dn):
        f.write(nm + "\t" + "\t".join(map(str, row.tolist())) + "\n")
T_write = time.time() - T2
print(f"n={n} L={L} parse={T_parse:.2f}s  gpu_relu={T_gpu:.2f}s  write={T_write:.2f}s  "
      f"peak_vram={torch.cuda.max_memory_allocated()/2**20:.0f}MiB", file=sys.stderr)
