// SPDX-License-Identifier: MIT
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif
  void gpufortrt_init();
  void gpufortrt_shutdown();
  
  void gpufortrt_data_start(gpufortrt_mapping_t* mappings,const int num_mappings);
  void gpufortrt_data_end(gpufortrt_mapping_t* mappings,const int num_mappings);
  //void gpufortrt_data_end(); // TODO discuss if internal stack shall be used instead of relying on frontend
                               // to decrement structured reference counters
  void gpufortrt_enter_exit_data(gpufortrt_mapping_t* mappings,
                       int num_mappings,
                       int async,
                       bool finalize);
  
  void* gpufortrt_use_device_b(void* hostptr,size_t num_bytes,
                     bool condition,bool if_present);
  void* gpufortrt_present_b(void* hostptr,size_t num_bytes
                  gpufortrt_counter_t ctr_to_update);
  void gpufortrt_dec_struct_refs_b(void* hostptr,int async);
  void gpufortrt_delete_b(void* hostptr,bool finalize);
  void* gpufortrt_no_create_b(void* hostptr,bool never_deallocate);
  void* gpufortrt_create_b(void* hostptr,size_t num_bytes,int async,bool never_deallocate,
                 gpufortrt_counter_t ctr_to_update);
  void* gpufortrt_copyin_b(void* hostptr,size_t num_bytes,int async,bool never_deallocate,
                 gpufortrt_counter_t ctr_to_update);
  void* gpufortrt_copyout_b(void* hostptr,size_t num_bytes,int async,bool never_deallocate,
                  gpufortrt_counter_t ctr_to_update);
  void* gpufortrt_copy_b(void* hostptr,size_t num_bytes,int async,bool never_deallocate
               gpufortrt_counter_t ctr_to_update);
  void gpufortrt_update_host_b(void* hostptr,bool condition,bool if_present,int async);
  void gpufortrt_update_device_b(void* hostptr,bool condition,bool if_present,int async);

  gpufortrt_queue_t gpufortrt_get_stream(int async);
} // extern "C"
