# Makefile for CUDA Ising Machine

NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_70 -lcurand

TARGET = ising_cuda
SRC = ising_cuda.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run

