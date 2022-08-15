# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
HIP_PLATFORM ?= amd
SUFFIX       ?= $(if $(HIP_PLATFORM),$(HIP_PLATFORM),amd)
LIBGPUFORTRT = libgpufortrt_$(SUFFIX).a

include rules.mk

CXXFLAGS ?= 
CXXFLAGS += -I./internal 

VPATH = ./:internal

# object files
include internal/make.inc

F_OBJ += ./gpufortrt_api.o \
         ./gpufortrt_api_core.o \
         ./gpufortrt_types.o
CXX_OBJ += ./gpufortrt_api.cpp.o

# targets

.PHONY: clean

$(LIBGPUFORTRT): $(CXX_OBJ) $(F_OBJ)
	ar -crs $@ $^ 

$(CXX_OBJ): %.cpp.o: %.cpp
	$(CXX) -c $< $(CXXFLAGS) -o $@

$(F_OBJ): %.o: %.f90
	$(FC) -c $< $(FCFLAGS)

clean:
	rm -f *.o *.mod *.a
