TARGET=main
OBJECTS=timer.o util.o mat_mul.o

CFLAGS=-std=c11 -O2 -fopenmp
LDFLAGS=-lm

all: $(TARGET)

$(TARGET): $(OBJECTS) mat_mul_ref.o

clean:
	rm -rf $(TARGET) $(OBJECTS)

validation: $(TARGET)
	./$(TARGET) -v -t 48 1008 48 16
	./$(TARGET) -v -t 49 80 176 336
	./$(TARGET) -v -t 28 656 320 464
	./$(TARGET) -v -t 48 224 720 752
	./$(TARGET) -v -t 2 576 464 672
	./$(TARGET) -v -t 64 704 496 736
	./$(TARGET) -v -t 28 48 960 720
	./$(TARGET) -v -t 48 1024 496 672
	./$(TARGET) -v -t 46 800 880 416
	./$(TARGET) -v -t 56 816 304 352

performance: $(TARGET)
	./$(TARGET) -v -t 16 -w 1 -n 1 4096 4096 4096
