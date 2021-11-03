include rules.mk

.PHONY: build codegen all clean

codegen:
	python3 codegen.py src/gpufort_acc_runtime.template.f90 -d 7

build:
	mkdir -p lib/
	mkdir -p include/
	make -C src/ libgpufort_acc.a
	mv src/*.mod include/
	mv src/libgpufort_acc.a lib/
	make -C src/ clean

all: | codegen build

clean:
	make -C src/ clean
	rm -rf include/ lib/
