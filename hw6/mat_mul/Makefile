TARGET=main
OBJECTS=timer.o util.o mat_mul.o

CC=gcc
CFLAGS=-std=c11 -O3 -Wall -fopenmp
LDLIBS=-lm
NSYS=/usr/local/cuda/bin/nsys
NCU=/usr/local/cuda/bin/ncu

SLURM_FLAGS=--nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --gres=gpu:1 --partition=shpc

# CUDA settings
OBJECTS+=cuda_util.o
CFLAGS+=-I/usr/local/cuda/include
LDFLAGS+=-L/usr/local/cuda/lib64
LDLIBS+=-lcudart -lstdc++

all: $(TARGET)

$(TARGET): $(OBJECTS) mat_mul_ref.o

mat_mul.o: mat_mul.cu
	/usr/local/cuda/bin/nvcc -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS) *.ncu-rep *.qdrep

validation: $(TARGET)
	salloc $(SLURM_FLAGS) mpirun ./$(TARGET) -v 896 1536 640
	salloc $(SLURM_FLAGS) mpirun ./$(TARGET) -v 1408 512 1792
	salloc $(SLURM_FLAGS) mpirun ./$(TARGET) -v 640 512 2048
	salloc $(SLURM_FLAGS) mpirun ./$(TARGET) -v 512 384 1664
	salloc $(SLURM_FLAGS) mpirun ./$(TARGET) -v 1152 768 896
	salloc $(SLURM_FLAGS) mpirun ./$(TARGET) -v 1024 384 1664
	salloc $(SLURM_FLAGS) mpirun ./$(TARGET) -v 1792 640 2048
	salloc $(SLURM_FLAGS) mpirun ./$(TARGET) -v 384 896 640
	salloc $(SLURM_FLAGS) mpirun ./$(TARGET) -v 256 1024 1920
	salloc $(SLURM_FLAGS) mpirun ./$(TARGET) -v 1792 768 1152

performance: $(TARGET)
	salloc $(SLURM_FLAGS) mpirun ./$(TARGET) -v -w 1 -n 1 8192 8192 8192

nsys : $(TARGET)
	salloc $(SLURM_FLAGS) mpirun $(NSYS) profile ./$(TARGET) -n 1 8192 8192 8192

ncu : $(TARGET)
	salloc $(SLURM_FLAGS) mpirun $(NCU) -o profile ./$(TARGET) -n 1 8192 8192 8192
