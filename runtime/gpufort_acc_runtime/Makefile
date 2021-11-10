include rules.mk

.PHONY: build codegen lib/$(LIBGPUFORT_ACC) clean

lib/$(LIBGPUFORT_ACC): | codegen build

codegen:
	python3 codegen.py src/gpufort_acc_runtime.template.f90 -d 7

build:
	mkdir -p lib/
	mkdir -p include/
	make -C src/ $(LIBGPUFORT_ACC)
	mv src/*.mod include/
	mv src/$(LIBGPUFORT_ACC) lib/
	make -C src/ clean

clean:
	make -C src/ clean
	rm -rf include/ lib/
