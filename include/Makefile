.PHONY: all headers modules

all: headers modules

headers:
	gpufort --create-gpufort-headers

modules:
	gpufort -K cudafor.f90

make clean:
	rm -rf *.gpufort_mod *-fort2hip.* *-gpufort.* *.h
