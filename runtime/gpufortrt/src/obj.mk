# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
F_OBJ += ./gpufortrt_types.o \
         ./gpufortrt_api.o \
         ./openacc.o
        #  ./gpufortrt_api_core.o \

CXX_OBJ += ./gpufortrt_types.cpp.o \
		   ./gpufortrt_api.cpp.o \
           ./openacc.cpp.o
