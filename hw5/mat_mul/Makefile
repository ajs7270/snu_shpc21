TARGET=main
OBJECTS=timer.o util.o mat_mul.o

CC=gcc
CFLAGS=-std=c11 -O3 -Wall -fopenmp
LDLIBS=-lm

SLURM_FLAGS=--nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --gres=gpu:1 --partition=shpc

# OpenCL settings
OBJECTS+=opencl_util.o
CFLAGS+=-I/usr/local/cuda/include
LDLIBS+=-L/usr/local/cuda/lib64 -lOpenCL

all: $(TARGET)

$(TARGET): $(OBJECTS) mat_mul_ref.o

clean:
	rm -rf $(TARGET) $(OBJECTS)

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
test: $(TARGET)
	salloc $(SLURM_FLAGS) mpirun ./$(TARGET) -v -w 1 -n 1 4096 4096 4096
