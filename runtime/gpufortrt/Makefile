# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
include rules.mk

CXXFLAGS ?= 
CXXFLAGS += -I./src -I./src/internal

VPATH = ./src:./src/internal

# gather objects
include src/internal/make.inc
include src/make.inc

.PHONY: build codegen lib/$(LIBGPUFORTRT) clean clean_all

# targets

build: lib/$(LIBGPUFORTRT)

lib/$(LIBGPUFORTRT): | codegen $(LIBGPUFORTRT)
	mkdir -p lib/
	mkdir -p include/
	mv *.mod include/
	mv $(LIBGPUFORTRT) lib/

$(LIBGPUFORTRT): $(CXX_OBJ) $(F_OBJ)
	ar -crs $@ $^ 

$(CXX_OBJ): %.cpp.o: %.cpp
	$(CXX) -c $< $(CXXFLAGS) -o $@

$(F_OBJ): %.o: %.f90
	$(FC) -c $< $(FCFLAGS)

codegen:
	python3 codegen.py src/gpufortrt_api.template.f90 -d 7

clean_all:
	rm -f *.o *.mod *.a
	rm -rf include/ lib/

clean:
	rm -f *.o *.mod *.a
