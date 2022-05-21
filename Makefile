.PHONY: clean_all gpufort_headers gpufort_sources gpufort_templates share/gpufort_sources lib/$(LIBGPUFORT) lib/$(LIBGPUFORT_ACC)

SUFFIX         = $(if $(HIP_PLATFORM),$(HIP_PLATFORM),amd)
LIBGPUFORT     = libgpufort_$(SUFFIX).a
LIBGPUFORT_ACC = libgpufort_acc_$(SUFFIX).a

GPUFORT_DIR     = .
GPUFORT_ACC_DIR = $(GPUFORT_DIR)/runtime/gpufort_acc_runtime

all: | gpufort_templates lib/$(LIBGPUFORT) lib/$(LIBGPUFORT_ACC) make_directories

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

lib/$(LIBGPUFORT_ACC): | gpufort_templates make_directories
	make -C $(GPUFORT_ACC_DIR)/ lib/$(LIBGPUFORT_ACC)
	mv $(GPUFORT_ACC_DIR)/lib/$(LIBGPUFORT_ACC)\
	    $(GPUFORT_DIR)/lib/
	mv $(GPUFORT_ACC_DIR)/include/*.mod\
	    $(GPUFORT_DIR)/include/$(SUFFIX)/
	#-mv $(GPUFORT_ACC_DIR)/include/*.h\
	#   $(GPUFORT_DIR)/include/$(SUFFIX)
	make -C $(GPUFORT_ACC_DIR)/ clean

make_directories:
	mkdir -p $(GPUFORT_DIR)/lib
	mkdir -p $(GPUFORT_DIR)/include/$(SUFFIX)

clean_all:
	make -C $(GPUFORT_DIR)/include clean_all
	make -C $(GPUFORT_DIR)/src     clean_all
	make -C $(GPUFORT_DIR)/python/gpufort/fort2x/templates clean_all
	make -C $(GPUFORT_DIR)/python/gpufort/fort2x/hip/templates clean_all
	rm -f lib/$(LIBGPUFORT) lib/$(LIBGPUFORT_ACC)
