include rules.mk

.PHONY: codegen build all clean

build: clean
	mkdir -p lib/
	mkdir -p include/
	make -C src/ libopenacc_gomp.a
	mv src/*.mod include/
	mv src/libopenacc_gomp.a lib/
	make -C src/ clean

codegen:
	cd codegen && python3 ./model.py
	cp codegen/openacc_gomp.f90 src/openacc_gomp.f90

all: | codegen build

clean:
	make -C src/ clean
	rm -rf include/ lib/
