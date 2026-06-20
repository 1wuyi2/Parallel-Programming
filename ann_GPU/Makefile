NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17

all: main_gpu

main_gpu: main_gpu.cu ann_ivf_gpu.cuh
	$(NVCC) $(NVCCFLAGS) main_gpu.cu -o main_gpu

run: main_gpu
	./main_gpu /anndata/ 2000 1024 50 10

clean:
	rm -f main_gpu