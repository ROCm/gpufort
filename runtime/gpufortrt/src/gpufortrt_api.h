// SPDX-License-Identifier: MIT
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif
  /**
   * \note: Enum values must match those of Fortran enumeration!
   * \note: Upper case first letter used because `delete` is C++ keyword.
   */
  // Core C++ API
  void init();
  void shutdown();
  
  void data_start(gpufortrt_mapping_t* mappings,const int num_mappings);
  void data_end();
  void enter_exit_data(gpufortrt_mapping_t* mappings,num_mappings,async,finalize);
  
  void* use_device_b(void* hostptr,size_t num_bytes,
                     bool condition,bool if_present);
  void* present_b(void* hostptr,size_t num_bytes
                  gpufortrt_counter_t ctr_to_update);
  void dec_struct_refs_b(void* hostptr,int async);
  void* no_create_b(void* hostptr);
  void* create_b(void* hostptr,size_t num_bytes,int async,bool never_deallocate,
                 gpufortrt_counter_t ctr_to_update);
  void delete_b(void* hostptr,finalize);
  void* copyin_b(void* hostptr,size_t num_bytes,int async,bool never_deallocate,
                 gpufortrt_counter_t ctr_to_update);
  void* copyout_b(void* hostptr,size_t num_bytes,int async,
                  gpufortrt_counter_t ctr_to_update);
  void* copy_b(void* hostptr,size_t num_bytes,int async,
               gpufortrt_counter_t ctr_to_update);
  void update_host_b(void* hostptr,bool condition,bool if_present,int async);
  void update_device_b(void* hostptr,bool condition,bool if_present,int async);
} // extern "C"