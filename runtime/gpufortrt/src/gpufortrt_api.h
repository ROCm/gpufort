// SPDX-License-Identifier: MIT
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "gpufortrt_types.h"

#ifdef __cplusplus
extern "C" {
#endif
  extern int gpufortrt_async_noval;

  void gpufortrt_init();
  void gpufortrt_shutdown();
  
  void gpufortrt_data_start(
          gpufortrt_mapping_t* mappings,
          int num_mappings);
  void gpufortrt_data_start_async(
          gpufortrt_mapping_t* mappings,
          int num_mappings,
          int async_arg);
  void gpufortrt_data_end();
  void gpufortrt_data_end_async(int async_arg);
  
  void gpufortrt_enter_exit_data(
          gpufortrt_mapping_t* mappings,
          int num_mappings,
          bool finalize);
  void gpufortrt_enter_exit_data_async(
          gpufortrt_mapping_t* mappings,
          int num_mappings,
          int async_arg,
          bool finalize);
  
  void* gpufortrt_use_device(
          void* hostptr,
          bool condition,
          bool if_present);

  // decrement dynamic reference counter
  
  void gpufortrt_delete(
          void* hostptr,
          size_t num_bytes);
  void gpufortrt_delete_finalize(
          void* hostptr,
          size_t num_bytes);
  void gpufortrt_delete_async(
          void* hostptr,
          size_t num_bytes,
          int async_arg);
  void gpufortrt_delete_finalize_async(
          void* hostptr,
          size_t num_bytes,
          int async_arg);
  
  void gpufortrt_copyout(
         void* hostptr,
         size_t num_bytes);
  void gpufortrt_copyout_async(
         void* hostptr,
         size_t num_bytes,
         int async_arg);
  void gpufortrt_copyout_finalize(
         void* hostptr,
         size_t num_bytes);
  void gpufortrt_copyout_finalize_async(
         void* hostptr,
         size_t num_bytes,
         int async_arg);
  
  // increment dynamic reference counter
  
  void* gpufortrt_present(
          void* hostptr,
          size_t num_bytes);
   
  void* gpufortrt_no_create(
          void* hostptr,
          size_t num_bytes);
  
  void* gpufortrt_create(
          void* hostptr,
          size_t num_bytes,
          bool never_deallocate);
  void* gpufortrt_create_async(
          void* hostptr,
          size_t num_bytes,
          int async_arg,
          bool never_deallocate);
  
  void* gpufortrt_copyin(
          void* hostptr,
          size_t num_bytes,
          bool never_deallocate);
  void* gpufortrt_copyin_async(
          void* hostptr,
          size_t num_bytes,
          int async_arg,
          bool never_deallocate);
  
  void* gpufortrt_copy(
         void* hostptr,
         size_t num_bytes,
         bool never_deallocate);
  void* gpufortrt_copy_async(
         void* hostptr,
         size_t num_bytes,
         int async_arg,
         bool never_deallocate);

  // other runtime calls

  void gpufortrt_update_host(
         void* hostptr,
         bool condition,
         bool if_present);
  void gpufortrt_update_self_async(
         void* hostptr,
         bool condition,
         bool if_present,
         int async_arg);
  void gpufortrt_update_self_section(
         void* hostptr,
         size_t num_bytes,
         bool condition,
         bool if_present);
  void gpufortrt_update_self_section_async(
         void* hostptr,
         size_t num_bytes,
         bool condition,
         bool if_present,
         int async_arg);

  void gpufortrt_update_device(
         void* hostptr,
         size_t num_bytes,
         bool condition,
         bool if_present);
  void gpufortrt_update_device_async(
         void* hostptr,
         bool condition,
         bool if_present,
         int async_arg);
  void gpufortrt_update_device_section(
         void* hostptr,
         size_t num_bytes,
         bool condition,
         bool if_present);
  void gpufortrt_update_device_section_async(
         void* hostptr,
         size_t num_bytes,
         bool condition,
         bool if_present,
         int async_arg);

  void gpufortrt_wait_all(bool condition);
  void gpufortrt_wait(int* wait_arg,
                      int num_wait,
                      bool condition);
  void gpufortrt_wait_async(int* wait_arg,int num_wait,
                            int* async_arg,int num_async,
                            bool condition);
  void gpufortrt_wait_all_async(int* async_arg,int num_async,
                                bool condition);

  gpufortrt_queue_t gpufortrt_get_stream(int async_arg);
 
  /** \return device pointer associated with `hostptr`, or nullptr.
   *  First searches through the structured region stack and then
   *  through the whole record list.
   *
   *  \note Does return a nullptr if `hostptr` is nullptr
   *  \note Does return `hostptr` if hostptr mapped via `no_create`
   *        and no record was found in the structured region stack,
   *        which stores mappings associated with structured data regions and
   *        compute constructs.
   *  \note Searches structured region stack first and then
   *        list of records, which also stores unstructured mappings.
   */
  void* gpufortrt_deviceptr(void* hostptr);

} // extern "C"
