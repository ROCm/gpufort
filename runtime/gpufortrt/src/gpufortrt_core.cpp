// SPDX-License-Identifier: MIT
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "gpufortrt_core.h"

#include "gpufortrt_auxiliary.h"

#include <string>
#include <sstream>

// global parameters, influenced by environment variables
size_t gpufortrt::internal::INITIAL_RECORDS_CAPACITY = 4096;
size_t gpufortrt::internal::INITIAL_QUEUE_RECORDS_CAPACITY = 128;
int gpufortrt::internal::BLOCK_SIZE = 32;
double gpufortrt::internal::REUSE_THRESHOLD = 0.9;
int gpufortrt::internal::NUM_REFS_TO_DEALLOCATE = -5;

size_t gpufortrt::internal::blocked_size(size_t num_bytes,size_t block_size) {
  return (((num_bytes)+block_size-1)/block_size) * block_size;
}

// global variables
bool gpufortrt::initialized = false;
size_t gpufortrt::num_records = 0; // TODO where is it now?
gpufortrt::internal::record_list_t                 gpufortrt::record_list;
gpufortrt::internal::queue_record_list_t gpufortrt::internal::queue_record_list;
