// SPDX-License-Identifier: MIT
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "gpufortrt_core.h"

#include "auxiliary.h"

#include <string>
#include <sstream>

// global parameters, influenced by environment variables
size_t gpufortrt::internal::INITIAL_RECORDS_CAPACITY = 4096;
size_t gpufortrt::internal::INITIAL_STRUCTURED_REGION_STACK_CAPACITY = 128;
size_t gpufortrt::internal::INITIAL_QUEUE_RECORDS_CAPACITY = 128;
int gpufortrt::internal::BLOCK_SIZE = 32;
double gpufortrt::internal::REUSE_THRESHOLD = 0.9;
int gpufortrt::internal::NUM_REFS_TO_DEALLOCATE = -5;

// global variables
bool gpufortrt::internal::initialized = false;
size_t gpufortrt::internal::num_records = 0; 
gpufortrt::internal::record_list_t gpufortrt::internal::record_list;
gpufortrt::internal::queue_record_list_t gpufortrt::internal::queue_record_list;
gpufortrt::internal::structured_region_stack_t gpufortrt::internal::structured_region_stack;
