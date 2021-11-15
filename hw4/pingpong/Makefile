TARGET=pingpong
OBJECTS=timer.o

CC=mpicc
CFLAGS=-std=c11 -O3 -Wall -fopenmp
LDFLAGS=-lm

SLURM_FLAGS=--nodes 2 --ntasks-per-node 1 --cpus-per-task=16 --partition=titan
MPIRUN_FLAGS=--bind-to none --mca btl openib,self --mca btl_openib_allow_ib true

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)

performance: $(TARGET)
	salloc $(SLURM_FLAGS) mpirun $(MPIRUN_FLAGS) ./$(TARGET)
