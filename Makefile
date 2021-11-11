.PHONY: clean_all gpufort_headers gpufort_sources lib/$(LIBGPUFORT) lib/$(LIBGPUFORT_ACC)

SUFFIX         = $(if $(HIP_PLATFORM),$(HIP_PLATFORM),amd)
LIBGPUFORT     = libgpufort_$(SUFFIX).a
LIBGPUFORT_ACC = libgpufort_acc_$(SUFFIX).a

GPUFORT_DIR     = $(shell gpufort --path)
GPUFORT_ACC_DIR = $(GPUFORT_DIR)/runtime/gpufort_acc_runtime

all: | gpufort_headers gpufort_sources lib/$(LIBGPUFORT) lib/$(LIBGPUFORT_ACC)

gpufort_headers:
	make -C $(GPUFORT_DIR)/include all

gpufort_sources:
	make -C $(GPUFORT_DIR)/src gpufort_sources

lib/$(LIBGPUFORT): gpufort_headers
	make -C $(GPUFORT_DIR)/src all
	mkdir -p $(GPUFORT_DIR)/lib
	mv $(GPUFORT_DIR)/src/$(LIBGPUFORT) $(GPUFORT_DIR)/lib
	mv $(GPUFORT_DIR)/src/*.mod $(GPUFORT_DIR)/include/$(SUFFIX)
	make -C $(GPUFORT_DIR)/src clean

lib/$(LIBGPUFORT_ACC):
	make -C $(GPUFORT_ACC_DIR)/ lib/$(LIBGPUFORT_ACC)
	mv $(GPUFORT_ACC_DIR)/lib/$(LIBGPUFORT_ACC)\
		$(GPUFORT_DIR)/lib/
	mkdir -p $(GPUFORT_DIR)/include/$(SUFFIX)
	mv $(GPUFORT_ACC_DIR)/include/*.mod\
		$(GPUFORT_DIR)/include/$(SUFFIX)
	#-mv $(GPUFORT_ACC_DIR)/include/*.h\
	#	$(GPUFORT_DIR)/include/$(SUFFIX)
	make -C $(GPUFORT_ACC_DIR)/ clean

clean_all:
	make -C $(GPUFORT_DIR)/include clean_all
	make -C $(GPUFORT_DIR)/src     clean_all
	rm -f lib/$(LIBGPUFORT) lib/$(LIBGPUFORT_ACC)
