include rules.mk

.PHONY: build codegen lib/$(LIBGPUFORTRT) clean

lib/$(LIBGPUFORTRT): | codegen build

codegen:
	python3 codegen.py src/gpufortrt_api.template.f90 -d 7

build:
	mkdir -p lib/
	mkdir -p include/
	make -C src/ $(LIBGPUFORTRT)
	mv src/*.mod include/
	mv src/$(LIBGPUFORTRT) lib/
	make -C src/ clean

clean:
	make -C src/ clean
	rm -rf include/ lib/
