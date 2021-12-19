CUDA_PATH=/usr/local/cuda

CC=mpicc
NVCC=$(CUDA_PATH)/bin/nvcc

CFLAGS=-O3 -Wall
LDFLAGS=-lm -lrt -lstdc++

OBJECTS=main.o qdbmp.o timer.o

all: seq parallel

seq: $(OBJECTS) facegen_seq.o
	$(CC) -o facegen_seq $^ $(LDFLAGS)

parallel: $(OBJECTS) facegen_parallel.o
	$(CC) -o facegen_parallel $^ $(LDFLAGS) -L$(CUDA_PATH)/lib64 -lcudart

facegen_parallel.o: facegen_parallel.cu
	$(NVCC) -ccbin $(CC) -c -o $@ $^

clean:
	rm -rf facegen_seq facegen_parallel $(OBJECTS) facegen_seq.o facegen_parallel.o

test_seq: facegen_seq
	salloc --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --partition=shpc mpirun ./facegen_seq network.bin input1.txt output1.txt output1.bmp

test : parallel
	salloc --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --gres=gpu:4 --partition=shpc mpirun ./facegen_parallel network.bin input3.txt output3.txt output3.bmp

