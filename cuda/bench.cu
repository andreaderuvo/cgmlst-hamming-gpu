// Same tiled kernel, two per-locus operations: ReLU network vs direct compare.
// Random clonal-ish ranks (n x L), full upper triangle, 3 timed runs each, and
// a check that both modes give the same matrix (stripe checksum).
//   ./bench <n> <L>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
extern "C" int rh_run(const uint16_t*, int, int, int, int, int,
                      void (*)(void*, int, int, const uint16_t*), void*, double*, char*, int);

struct Sum { int n; uint64_t h; };
static void cb(void* ctx, int r0, int rows, const uint16_t* s) {
    Sum* c = (Sum*)ctx;
    for (int r = 0; r < rows; r++)
        for (int j = r0 + r + 1; j < c->n; j++)
            c->h = c->h * 1000003u + s[(size_t)r * c->n + j];
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <n> <L>\n", argv[0]); return 1; }
    int n = atoi(argv[1]), L = atoi(argv[2]);
    std::vector<uint16_t> X((size_t)n * L);
    std::mt19937 g(1);
    std::vector<uint16_t> F((size_t)50 * L);
    for (auto& v : F) v = 1 + g() % 200;
    for (int i = 0; i < n; i++) {
        int f = g() % 50;
        for (int k = 0; k < L; k++) {
            uint32_t r = g() % 100;
            X[(size_t)i * L + k] = r < 2 ? 0 : (r < 5 ? 201 + g() % 4000 : F[(size_t)f * L + k]);
        }
    }
    char err[256];
    uint64_t h[2];
    for (int relu = 1; relu >= 0; relu--) {
        printf("%-8s", relu ? "relu" : "compare");
        for (int r = 0; r < 3; r++) {
            Sum s{n, 0};
            double secs;
            if (rh_run(X.data(), n, L, 0, 0, relu, cb, &s, &secs, err, 256)) { puts(err); return 1; }
            printf("  %.2f s", secs);
            h[relu] = s.h;
        }
        printf("\n");
    }
    printf("identical matrices: %s\n", h[0] == h[1] ? "yes" : "NO");
    return h[0] == h[1] ? 0 : 1;
}
