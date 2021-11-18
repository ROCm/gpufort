CFLAGS = -g -ggdb

vector-add:
	hipcc $(shell gpufort --cpp_config) $(CFLAGS) -c main-kernels.hip.cpp -o main-kernels.hip.o
	hipfc $(shell gpufort --gfortran_config) main-kernels.hip.o vector-add.f90 -o vector-add $(shell gpufort --ldflags)

clean:
	rm -f *.mod *.o vector-add
