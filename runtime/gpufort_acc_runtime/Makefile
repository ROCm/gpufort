include rules.mk

.PHONY: build codegen all clean

build:
	mkdir -p lib/
	mkdir -p include/
	make -C src/ libgpufort_acc.a
	mv src/*.mod include/
	mv src/libgpufort_acc.a lib/
	make -C src/ clean

codegen:
	cd codegen && python3 ./model.py
	cp codegen/gpufort_acc_runtime.f90 src/gpufort_acc_runtime.f90

all: | codegen build

clean:
	make -C src/ clean
	rm -rf include/ lib/
