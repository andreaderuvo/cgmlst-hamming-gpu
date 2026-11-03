"""
Hamming distance (cgMLST-style) as an exact, hand-wired ReLU network.
Input per locus: a, b = allele IDs (integers >= 0, 0 = missing, as in chewBBACA).
Output: number of loci where both alleles are present and differ.  No if, no ==, only ReLU + linear.
"""
import numpy as np

relu = lambda z: np.maximum(z, 0)

# ---- weights, written by hand (a "compiled" MLP) ----
# layer 1 : input [a, b, 1] -> 6 hidden units
W1 = np.array([
    [ 1, -1, 0],   # h0 = relu(a-b)
    [-1,  1, 0],   # h1 = relu(b-a)        h0+h1 = |a-b|
    [ 1, -1,-1],   # h2 = relu(a-b-1)
    [-1,  1,-1],   # h3 = relu(b-a-1)      h2+h3 = relu(|a-b|-1)
    [ 1,  0,-1],   # h4 = relu(a-1)
    [ 0,  1,-1],   # h5 = relu(b-1)
], dtype=float)
# layer 2 (linear): combine into  diff, pa, pb
#   diff = min(1,|a-b|) = |a-b| - relu(|a-b|-1)        (exact for integers)
#   pa   = min(1,a)     = a - relu(a-1)   -> needs raw a, so pass it through
W2 = np.array([
    [1, 1, -1, -1, 0, 0, 0, 0],   # diff
    [0, 0,  0,  0,-1, 0, 1, 0],   # pa = a - h4
    [0, 0,  0,  0, 0,-1, 0, 1],   # pb = b - h5
], dtype=float)
# layer 3 : AND of three {0,1} bits = relu(diff + pa + pb - 2)
W3 = np.array([[1, 1, 1]], dtype=float); b3 = -2.0

def locus_net(a, b):
    """a, b: arrays (..., L) of allele IDs. Returns per-locus mismatch bit."""
    a, b = np.broadcast_arrays(a, b)
    x = np.stack([a, b, np.ones_like(a)], -1).astype(float)       # (..., L, 3)
    h = relu(x @ W1.T)                                            # (..., L, 6)
    h = np.concatenate([h, x[..., :2]], -1)                       # skip-connection a, b (relu(a)=a since a>=0)
    z = h @ W2.T                                                  # (..., L, 3)
    return relu(z @ W3.T + b3)[..., 0]                            # (..., L)

def hamming_relu(P):
    """P: (n, L) profiles -> (n, n) distance matrix, sum-pooling over loci."""
    return locus_net(P[:, None, :], P[None, :, :]).sum(-1).astype(int)

def hamming_ref(P):
    A, B = P[:, None, :], P[None, :, :]
    return ((A != B) & (A > 0) & (B > 0)).sum(-1)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for trial in range(20):
        n, L = rng.integers(2, 60), rng.integers(1, 3000)
        P = rng.integers(0, rng.integers(2, 50000), size=(n, L))
        P[rng.random(P.shape) < 0.05] = 0                         # missing
        assert np.array_equal(hamming_relu(P), hamming_ref(P)), trial
    # edge cases: equal, adjacent IDs, huge IDs, missing on either side
    a = np.array([5, 5, 6, 0, 7, 0, 10**9]); b = np.array([5, 6, 5, 3, 0, 0, 1])
    print("per-locus bits:", locus_net(a, b).astype(int), "expected [0 1 1 0 0 0 1]")
    print("20 random matrices: ReLU net == reference  OK")
