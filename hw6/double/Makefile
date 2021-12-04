TARGET=main
OBJECTS=timer.o util.o vec_add.o

CC=mpicc
CFLAGS=-std=c11 -O3 -Wall -fopenmp
LDLIBS=-lm

SLURM_FLAGS=--nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --gres=gpu:1 --partition=shpc

# CUDA settings
OBJECTS+=cuda_util.o
CFLAGS+=-I/usr/local/cuda/include
LDFLAGS+=-L/usr/local/cuda/lib64
LDLIBS+=-lcudart -lstdc++

all: $(TARGET)

$(TARGET): $(OBJECTS)

vec_add.o: vec_add.cu
	/usr/local/cuda/bin/nvcc -c -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)

test: $(TARGET)
	salloc $(SLURM_FLAGS) mpirun ./$(TARGET) 209715200
