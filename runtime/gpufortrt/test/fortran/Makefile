# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
HIPCC ?= hipcc 
HIPCC_CFLAGS = $(shell gpufort --print-cpp-config)

HIPFC ?= hipfc
HIPFC_CFLAGS ?= $(shell gpufort --print-gfortran-config)

LDFLAGS = $(shell gpufort --print-acc-ldflags) -lgfortran

# sources
TEST_SRC      = \
				./test_structured_region_00.f90 \
				./test_structured_region_01.f90 \
				./test_structured_region_02.f90 \
				./test_structured_region_03.f90 \
				./test_unstructured_region_00.f90 \
				./test_unstructured_region_01.f90 \
				./test_unstructured_region_02.f90 \
				./test_unstructured_region_03.f90 \
				./test_mixed_regions_00.f90 \
				./test_declare_00.f90 \
				./test_declare_01.f90 \
				./test_no_create_clause_00.f90
TEST_NAME     = $(TEST_SRC:.f90=)
TEST_CONV_SRC = $(TEST_SRC:.f90=.f90-gpufort.f08)

# targets
.PHONY: all clean $(TEST_NAME)

all: $(TEST_NAME)

$(TEST_NAME): %: %.f90-gpufort.f08
	$(HIPCC) $(HIPCC_CFLAGS) -c $@.f90-gpufort.cpp
	$(HIPFC) $^ -o $@ $@.f90-gpufort.o $(HIPFC_CFLAGS) $(LDFLAGS)

$(TEST_CONV_SRC): %-gpufort.f08: %
	gpufort --wrap $^ --dest hipgpufort

clean:
	rm -rf *-gpufort.* *-fort2*.* *.o *.mod gpufort*.h $(TEST_NAME) *.gpufort_mod *-linemaps-*.json log/
