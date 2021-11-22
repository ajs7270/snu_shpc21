TARGET=main
OBJECTS=timer.o util.o vec_add.o

CC=gcc
CFLAGS=-std=c11 -O3 -Wall -fopenmp
LDLIBS=-lm

SLURM_FLAGS=--nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --gres=gpu:4 --partition=shpc

# OpenCL settings
OBJECTS+=opencl_util.o
CFLAGS+=-I/usr/local/cuda/include
LDLIBS+=-L/usr/local/cuda/lib64 -lOpenCL

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)

test: $(TARGET)
	salloc $(SLURM_FLAGS) mpirun ./$(TARGET) 209715200
