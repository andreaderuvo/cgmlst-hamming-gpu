// Direct-compare kernel (../cuda/hamming.cu) on a real rank file; prints time + checksum.
#include <cstdint>
#include <cstdio>
#include <vector>
extern "C" int rh_run(const uint16_t*, int, int, int, int, int,
                      void (*)(void*, int, int, const uint16_t*), void*, double*, char*, int);
struct Sum { int n; uint64_t h; };
static void cb(void* ctx, int r0, int rows, const uint16_t* s) { return;
    Sum* c = (Sum*)ctx;
    for (int r = 0; r < rows; r++)
        for (int j = r0 + r + 1; j < c->n; j++) c->h = c->h * 1000003u + s[(size_t)r * c->n + j];
}
int main(int argc, char** argv) {
    FILE* f = fopen(argv[1], "rb"); int nl[2]; fread(nl, 4, 2, f);
    std::vector<uint16_t> X((size_t)nl[0] * nl[1]); fread(X.data(), 2, X.size(), f); fclose(f);
    char err[256];
    for (int r = 0; r < 3; r++) {
        Sum s{nl[0], 0}; double secs;
        if (rh_run(X.data(), nl[0], nl[1], 0, 0, 0, cb, &s, &secs, err, 256)) { puts(err); return 1; }
        printf("compare kernel n=%d L=%d: %.2f s (pipeline incl. checksum on host)  checksum=%llx\n",
               nl[0], nl[1], secs, (unsigned long long)s.h);
    }
}
