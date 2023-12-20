// SPDX-License-Identifier: MIT
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "gpufortrt_core.h"

#include "auxiliary.h"

#include <string>
#include <sstream>

// global parameters, influenced by environment variables
std::size_t gpufortrt::internal::INITIAL_RECORDS_CAPACITY = 4096;
std::size_t gpufortrt::internal::INITIAL_STRUCTURED_REGION_STACK_CAPACITY = 128;
std::size_t gpufortrt::internal::INITIAL_QUEUE_RECORDS_CAPACITY = 128;
int gpufortrt::internal::BLOCK_SIZE = 32;
double gpufortrt::internal::REUSE_THRESHOLD = 0.9;
int gpufortrt::internal::NUM_REFS_TO_DEALLOCATE = -5;

// global variables
int gpufortrt::internal::default_async_arg = gpufortrt_async_noval;

bool gpufortrt::internal::initialized = false;

std::size_t gpufortrt::internal::num_records = 0; 

gpufortrt::internal::record_list_t gpufortrt::internal::record_list;

gpufortrt::internal::queue_record_list_t gpufortrt::internal::queue_record_list;

gpufortrt::internal::structured_region_stack_t gpufortrt::internal::structured_region_stack;

bool gpufortrt::internal::implies_allocate_device_buffer(
                            const gpufortrt_map_kind_t map_kind,
                            const gpufortrt_counter_t ctr) {
    return 
         map_kind == gpufortrt_map_kind_create
      || map_kind == gpufortrt_map_kind_copyin
      || map_kind == gpufortrt_map_kind_copy
      || (    map_kind == gpufortrt_map_kind_copyout
           && ctr == gpufortrt_counter_structured );
}
bool gpufortrt::internal::implies_copy_to_device(const gpufortrt_map_kind_t map_kind) {
  return 
       map_kind == gpufortrt_map_kind_copyin
    || map_kind == gpufortrt_map_kind_copy;
}

bool gpufortrt::internal::implies_copy_to_host(const gpufortrt_map_kind_t map_kind) {
  return 
       map_kind == gpufortrt_map_kind_copy
    || map_kind == gpufortrt_map_kind_copyout;
}
