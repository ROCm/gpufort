.PHONY: clean_all gpufort_headers gpufort_sources gpufort_templates share/gpufort_sources lib/$(LIBGPUFORT) lib/$(LIBGPUFORTRT)

SUFFIX         = $(if $(HIP_PLATFORM),$(HIP_PLATFORM),amd)
LIBGPUFORT     = libgpufort_$(SUFFIX).a
LIBGPUFORTRT = libgpufortrt_$(SUFFIX).a

GPUFORT_DIR     = .
GPUFORTRT_DIR = $(GPUFORT_DIR)/runtime/gpufortrt

all: | gpufort_templates lib/$(LIBGPUFORT) lib/$(LIBGPUFORTRT) make_directories

gpufort_templates:
	make -C $(GPUFORT_DIR)/python/gpufort/fort2x/templates all
	make -C $(GPUFORT_DIR)/python/gpufort/fort2x/hip/templates all

gpufort_headers:
	make -C $(GPUFORT_DIR)/include all

gpufort_sources:
	make -C $(GPUFORT_DIR)/src gpufort_sources

share/gpufort_sources: gpufort_headers gpufort_sources
	cp src/gpufort.f03 share/gpufort_sources
	cp src/gpufort_array.cpp share/gpufort_sources
	cp src/gpufort_array.f03 share/gpufort_sources
	cp include/gpufort.h share/gpufort_sources
	cp include/gpufort_array.h share/gpufort_sources
	cp include/gpufort_reduction.h share/gpufort_sources

lib/$(LIBGPUFORT): | gpufort_templates gpufort_headers gpufort_sources make_directories
	make -C $(GPUFORT_DIR)/src $(LIBGPUFORT)
	mv $(GPUFORT_DIR)/src/$(LIBGPUFORT) $(GPUFORT_DIR)/lib
	mv $(GPUFORT_DIR)/src/*.mod $(GPUFORT_DIR)/include/$(SUFFIX)/
	make -C $(GPUFORT_DIR)/src clean

lib/$(LIBGPUFORTRT): | gpufort_templates make_directories
	make -C $(GPUFORTRT_DIR)/ clean_all build clean
	mv $(GPUFORTRT_DIR)/lib/$(LIBGPUFORTRT)\
	    $(GPUFORT_DIR)/lib/
	mv $(GPUFORTRT_DIR)/include/*.mod\
	    $(GPUFORT_DIR)/include/$(SUFFIX)/
	#-mv $(GPUFORTRT_DIR)/include/*.h\
	#   $(GPUFORT_DIR)/include/$(SUFFIX)
	make -C $(GPUFORTRT_DIR)/ clean

make_directories:
	mkdir -p $(GPUFORT_DIR)/lib
	mkdir -p $(GPUFORT_DIR)/include/$(SUFFIX)

clean_all:
	make -C $(GPUFORT_DIR)/include clean_all
	make -C $(GPUFORT_DIR)/src     clean_all
	make -C $(GPUFORT_DIR)/python/gpufort/fort2x/templates clean_all
	make -C $(GPUFORT_DIR)/python/gpufort/fort2x/hip/templates clean_all
	rm -f lib/$(LIBGPUFORT) lib/$(LIBGPUFORTRT)
