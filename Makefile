NVCC ?= /usr/local/cuda/bin/nvcc
ARCH ?= native

bench: cuda/hamming.cu cuda/bench.cu
	$(NVCC) -O3 -arch=$(ARCH) $^ -o $@

clean:
	rm -f bench
