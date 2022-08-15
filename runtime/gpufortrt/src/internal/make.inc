# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
CXX_SRC = ./queue_record_t.cpp \
          ./structured_region_stack_t.cpp \
          ./queue_record_list_t.cpp \
          ./record_list_t.cpp \
          ./record_t.cpp \
          ./auxiliary.cpp \
          ./gpufortrt_core.cpp

CXX_OBJ += $(CXX_SRC:.cpp=.cpp.o) 
