# 50k-scale: row stripes so VRAM stays < ~1.5 GB; verifies random rows against numpy reference
import sys, time, numpy as np, torch, triton, triton.language as tl
from relu_triton import load

@triton.jit
def stripe_kernel(X, D, n, L, r0, rows, sx, sd, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pi, pj = tl.program_id(0), tl.program_id(1)
    rm = r0 + pi * BM + tl.arange(0, BM); rn = pj * BN + tl.arange(0, BN)
    if pj * BN + BN <= r0 + pi * BM:            # tile entirely below diagonal: skip (mirrored on host)
        return
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    mr = rm < r0 + rows
    for k0 in range(0, L, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rm[:, None] * sx + rk[None, :], mask=mr[:, None] & (rk[None, :] < L), other=0).to(tl.float32)[:, None, :]
        y = tl.load(X + rn[:, None] * sx + rk[None, :], mask=(rn[:, None] < n) & (rk[None, :] < L), other=0).to(tl.float32)[None, :, :]
        d = x - y
        diff = tl.maximum(d, 0.) + tl.maximum(-d, 0.) - tl.maximum(d - 1., 0.) - tl.maximum(-d - 1., 0.)
        pa = x - tl.maximum(x - 1., 0.); pb = y - tl.maximum(y - 1., 0.)
        acc += tl.sum(tl.maximum(diff + pa + pb - 2., 0.), axis=2)
    tl.store(D + (rm[:, None] - r0) * sd + rn[None, :], acc.to(tl.int16), mask=mr[:, None] & (rn[None, :] < n))

inp = sys.argv[1]
import os
if os.path.exists(inp + ".npy"): codes = np.load(inp + ".npy")
else:
    t = time.time(); names, codes = load(inp); print(f"parse {time.time()-t:.1f}s", file=sys.stderr); np.save(inp + ".npy", codes)
assert codes.max() < 32767
n, L = codes.shape
X = torch.from_numpy(codes.astype(np.int16)).cuda()   # 500 MB
S = 1024                                           # stripe rows -> 4096*50k*2B = 400 MB on GPU
buf = torch.empty((S, n), dtype=torch.int16, device="cuda")
Dh = np.empty((n, n), dtype=np.int16)
BM = BN = 32; BK = 8
stripe_kernel[(1, 1)](X, buf, n, L, 0, 32, X.stride(0), buf.stride(0), BM=BM, BN=BN, BK=BK, num_warps=4)
torch.cuda.synchronize(); t = time.time(); tk = 0.
for r0 in range(0, n, S):
    rows = min(S, n - r0)
    k = time.time()
    stripe_kernel[(triton.cdiv(rows, BM), triton.cdiv(n, BN))](X, buf, n, L, r0, rows, X.stride(0), buf.stride(0),
                                                                BM=BM, BN=BN, BK=BK, num_warps=4)
    torch.cuda.synchronize(); tk += time.time() - k
    Dh[r0:r0+rows, r0:] = buf[:rows, r0:].cpu().numpy()
iu = np.triu_indices(n, 1)
Dh.T[iu] = Dh[iu]            # mirror upper -> lower
print(f"n={n} L={L} kernel={tk:.1f}s total_gpu+copy+mirror={time.time()-t:.1f}s peak_vram={torch.cuda.max_memory_allocated()/2**20:.0f}MiB", file=sys.stderr)
# verification: 200 random rows vs numpy reference (!= with missing mask)
rng = np.random.default_rng(7); C = codes
for i in rng.choice(n, 200, replace=False):
    ref = ((C[i] != C) & (C[i] > 0) & (C > 0)).sum(1)
    assert np.array_equal(Dh[i], ref), i
print("200 random rows == numpy reference OK", file=sys.stderr)
np.save(sys.argv[2], Dh[:2000, :2000])
