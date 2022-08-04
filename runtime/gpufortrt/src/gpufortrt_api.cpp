// SPDX-License-Identifier: MIT
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "gpufortrt_core.h"

#include <string>
#include <sstream>

#include <hip/hip_runtime_api.h>

#include "gpufortrt_auxiliary.h"

gpufortrt_async_noval = -1;

void gpufortrt_mapping_init(
    gpufortrt_mapping_t* mapping,
    void* hostptr,
    size_t num_bytes,
    gpufortrt_map_kind_t map_kind,
    bool never_deallocate) {
  mapping->hostptr    = hostptr;
  mapping->num_bytes  = num_bytes;
  mapping->map_kind   = map_kind;
  mapping->never_deallocate = never_deallocate;
}

void gpufortrt_init() {
  if ( gpufortrt::internal::initialized ) {
    throw std::invalid_argument("init: runtime has already been initialized");
  else {
    set_from_environment(gpufortrt::internal::LOG_LEVEL,"GPUFORTRT_LOG_LEVEL");
    set_from_environment(gpufortrt::internal::INITIAL_RECORDS_CAPACITY,"GPUFORTRT_INITIAL_RECORDS_CAPACITY");
    set_from_environment(gpufortrt::internal::INITIAL_QUEUE_RECORDS_CAPACITY,"GPUFORTRT_INITIAL_QUEUE_RECORDS_CAPACITY");
    set_from_environment(gpufortrt::internal::BLOCK_SIZE,"GPUFORTRT_BLOCK_SIZE");
    set_from_environment(gpufortrt::internal::REUSE_THRESHOLD,"GPUFORTRT_REUSE_THRESHOLD");
    set_from_environment(gpufortrt::internal::NUM_REFS_TO_DEALLOCATE,"GPUFORTRT_NUM_REFS_TO_DEALLOCATE");
    gpufortrt::internal::record_list.initialize();
    gpufortrt::internal::record_queue_list.initialize();
    gpufortrt::internal::initialized = true;
    //
    LOG_INFO(1,"initialized runtime")
    LOG_INFO(1,"GPUFORTRT_LOG_LEVEL=" << gpufortrt::internal::LOG_LEVEL)
    LOG_INFO(1,"GPUFORTRT_INITIAL_QUEUE_RECORDS_CAPACITY=" << gpufortrt::internal::INITIAL_QUEUE_RECORDS_CAPACITY)
    LOG_INFO(1,"GPUFORTRT_INITIAL_RECORDS_CAPACITY=" << gpufortrt::internal::INITIAL_RECORDS_CAPACITY)
    LOG_INFO(1,"GPUFORTRT_BLOCK_SIZE=" << gpufortrt::internal::BLOCK_SIZE)
    LOG_INFO(1,"GPUFORTRT_REUSE_THRESHOLD=" << gpufortrt::internal::REUSE_THRESHOLD)
    LOG_INFO(1,"GPUFORTRT_NUM_REFS_TO_DEALLOCATE=" << gpufortrt::internal::NUM_REFS_TO_DEALLOCATE)
  }
}

void gpufortrt_shutdown() {
  if ( !gpufortrt::internal::initialized ) {
    LOG_ERROR("gpufortrt_shutdown: runtime has not been initialized")
  }
  gpufortrt::record_list.destroy();
  gpufortrt::internal::queue_record_list.destroy();
  LOG_INFO(1,"terminated runtime")
}

namespace {
  void* run_use_increment_action(
      void*& hostptr,
      size_t num_bytes,
      gpufortrt_map_kind_t map_kind,
      int async,
      bool never_deallocate,
      gpufortrt_counter_t ctr_to_update) {
    if ( ! initialized_ ) gpufortrt_init();
    if ( hostptr == nullptr) {
      #ifndef NULLPTR_MEANS_NOOP
      std::stringstream ss;
      ss << "gpufortrt_" << map_kind << "_b: " << "hostptr must not be null"; 
      throw std::invalid_argument(ss.str());
      #endif
      return nullptr;
    } else {
      bool blocking_copy = async < 0; // TODO nullptr vs blocking
      gpufortrt_queue_t queue = gpufortrt::internal::queue_record_list.use_create_queue(async);
      
      size_t loc = gpufortrt::internal::record_list.use_increment_record(
        hostptr,
        num_bytes,&
        map_kind,&
        ctr_to_update,
        blocking_copy,
        queue,
        never_deallocate);
      return gpufortrt::internal::record_list.records[loc].deviceptr;
    }
  }

  void apply_mappings(gpufortrt_mapping_t* mappings,
                      int num_mappings,
                      gpufortrt_counter_t ctr_to_update,
                      int async,bool finalize) {
    for (int i = 0; i < num_mappings; i++) {
       auto& mapping = mappings[i];
       switch (mapping.map_kind) {
         case gpufortrt_map_kind_dec_struct_refs:
            gpufortrt_dec_struct_refs_b(mapping.hostptr,async);
            break;
         case gpufortrt_map_kind_delete:
            gpufortrt_delete_b(mapping.hostptr,finalize);
            break;
         case gpufortrt_map_kind_present:
         case gpufortrt_map_kind_create:
         case gpufortrt_map_kind_no_create:
         case gpufortrt_map_kind_copyin:
         case gpufortrt_map_kind_copyout:
         case gpufortrt_map_kind_copy:
            ::run_use_increment_action(
              mapping.hostptr,
              mapping.num_bytes,
              mapping.map_kind
              async,
              mapping.never_deallocate, 
              ctr_to_update);
            break;
         default: std::invalid_argument("apply_mappings: invalid map_kind"); break;
       }
    }
  }

  void wait(int* wait_arg,int num_wait) {
    for (int i = 0; i < num_wait; i++) {
      auto& queue_record = gpufortrt::internal::queue_record_list.use_create_queue(wait_arg[i]]);
      HIP_CHECK(hipStreamSynchronize(queue_record.queue)) // TODO backend specific, externalize
    }
  }

  void wait_async(int* wait_arg,int num_wait,
                  int* async,int num_async) {
    for (int i = 0; i < num_wait; i++) {
      hipEvent_t event;// TODO backend specific, externalize
      HIP_CHECK(hipEventCreateWithFlags(&event,hipEventDisableTiming))// TODO backend specific, externalize
      auto& queue_record = gpufortrt::internal::queue_record_list.use_create_queue(wait_arg[i]]);
      HIP_CHECK(hipEventRecord(event,record.queue))// TODO backend specific, externalize
      for (int j = 0; j < num_async; j++) {
        auto& queue_record_async = gpufortrt::internal::queue_record_list.use_create_queue(async[i]]);
        HIP_CHECK(hipStreamWaitEvent(queue_record_async.queue,event);// TODO backend specific, externalize
      }
    }
  }
}

void gpufortrt_data_start(gpufortrt_mapping_t* mappings,int num_mappings) {
  apply_mappings(mappings,
                 num_mappings,
                 gpufortrt_counter_t_structured,
                 -1/*async*/,false/*finalize*/);
}

void gpufortrt_data_end(gpufortrt_mapping_t* mappings,int num_mappings) {
  gpufortrt_data_start(mappings,num_mappings);
}

void gpufortrt_enter_exit_data(gpufortrt_mapping_t* mappings,
                               int num_mappings,
                               int async,
                               bool finalize) {
  apply_mappings(mappings,
                 num_mappings,
                 gpufortrt_counter_t_dynamic,
                 async,finalize);
}

void* gpufortrt_use_device_b(void* hostptr,size_t num_bytes,
                              bool condition,bool if_present) {
  void* resultptr = hostptr;
  if ( hostptr == nullptr ) {
     return nullptr;
  } else if ( condition ) {
    size_t loc = record_list_find_record_(hostptr,success);
    // TODO already detect a suitable candidate in find_record on-the-fly
    if ( success ) {
      auto& record = gpufortrt::internal::record_list.records[loc];
      size_t offset_bytes;
      bool fits = record.is_subarray(hostptr,num_bytes,offset_bytes/*inout*/); // TODO might not fit, i.e. only subarray
                                                                      // might have been mapped before
      if ( !fits ) LOG_ERROR("gpufortrt_use_device_b: input data region overlaps with previously mapped data but is no subset of it") 
      return static_cast<void*>(static_cast<char*>(record%deviceptr) + offset_bytes);
    } else if ( if_present ) {
      return hostptr;
    } else {
      LOG_ERROR("gpufortrt_use_device_b: did not find record for hostptr " << hostptr)
    }
  }
}

void  gpufortrt_dec_struct_refs_b(void* hostptr,int async) {
  if ( !initialized_ ) LOG_ERROR("gpufortrt_delete_b: runtime not initialized")
  if ( hostptr == nullptr ) {
#ifndef NULLPTR_MEANS_NOOP
    throw std::invalid_argument("gpufortrt_delete_b: hostptr is nullptr");
#endif
    return
  else {
    bool blocking_copy = async < 0; // TODO nullptr vs blocking
    gpufortrt_queue_t queue = gpufortrt::internal::queue_record_list.use_create_queue(async);
    
    gpufortrt::internal::record_list.decrement_release_record(
      hostptr,
      gpufortrt_counter_t_structured,
      veto_copy_to_host,
      blocking_copy,
      queue,
      false);
  }
}

void gpufortrt_delete_b(void* hostptr,bool finalize) {
  if ( !initialized_ ) LOG_ERROR("gpufortrt_delete_b: runtime not initialized")
  if ( hostptr == nullptr ) {
#ifndef NULLPTR_MEANS_NOOP
    throw std::invalid_argument("gpufortrt_delete_b: hostptr is nullptr");
#endif
    return
  else {
    gpufortrt::internal::record_list.decrement_release_record(
      hostptr,
      gpufortrt_counter_t_dynamic,
      veto_copy_to_host,
      blocking_copy,
      queue,
      finalize);
  }
}

void* gpufortrt_present_b(void* hostptr,size_t num_bytes,
                          gpufortrt_counter_t ctr_to_update) {
  return ::run_use_increment_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_present,
    async,
    false, 
    ctr_to_update);
}

void* gpufortrt_no_create_b(void* hostptr,size_t num_bytes,bool never_deallocate,
                            gpufortrt_counter_t ctr_to_update) {
  return ::run_use_increment_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_no_create,
    async,
    never_deallocate, 
    ctr_to_update);
}



void* gpufortrt_create_b(void* hostptr,size_t num_bytes,bool never_deallocate,
                         gpufortrt_counter_t ctr_to_update) {
  return ::run_use_increment_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_copyin,
    async,
    never_deallocate, 
    ctr_to_update);
}

void* gpufortrt_copyin_b(void* hostptr,size_t num_bytes,int async,bool never_deallocate,
                          gpufortrt_counter_t ctr_to_update) {
  return ::run_use_increment_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_copyin,
    async,
    never_deallocate, 
    ctr_to_update);
}

void* gpufortrt_copyout_b(void* hostptr,size_t num_bytes,int async,bool never_deallocate,
                           gpufortrt_counter_t ctr_to_update) {
  return ::run_use_increment_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_copyout,
    async,
    never_deallocate,
    ctr_to_update);
}

void* gpufortrt_copy_b(void* hostptr,size_t num_bytes,int async,bool never_deallocate,
                       gpufortrt_counter_t ctr_to_update) {
  return ::run_use_increment_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_copy,
    async,
    never_deallocate,
    ctr_to_update);
}

void gpufortrt_update_host_b(
    void* hostptr,
    bool condition,
    bool if_present,
    int async) {
  if ( !initialized_ ) LOG_ERROR("gpufortrt_update_host_b: runtime not initialized")
  if ( hostptr == nullptr ) {
#if NULLPTR_MEANS_NOOP
    throw std::invalid_argument("gpufortrt_update_host_b: hostptr is nullptr")
#endif
    return;
  }
  if ( condition ) {
    size_t loc = gpufortrt::internal::record_list.find_record(hostptr,success);
    bool blocking_copy = async < 0; // TODO nullptr vs blocking
   
    if ( !success && !if_present ) { 
      LOG_ERROR("gpufortrt_update_host_b: no deviceptr found for hostptr")
    } else if ( success && blocking_copy ) {
      auto& record = gpufortrt::internal::record_list.records[loc];
      record.copy_to_host(.false.,nullptr);
    } else if ( success ) {
      auto& record = gpufortrt::internal::record_list.records[loc];
      gpufortrt_queue_t queue = gpufortrt::internal::queue_record_list.use_create_queue(async);
      record.copy_to_host(blocking_copy,queue);
    }
  }
}

void gpufortrt_update_device_b(
    void* hostptr,
    bool condition,
    bool if_present,
    int async) {
  if ( !initialized_ ) LOG_ERROR("gpufortrt_update_device_b: runtime not initialized")
  if ( deviceptr == nullptr ) {
#if NULLPTR_MEANS_NOOP
    throw std::invalid_argument("gpufortrt_update_device_b: deviceptr is nullptr")
#endif
    return;
  }
  if ( condition ) {
    size_t loc = gpufortrt::internal::record_list.find_record(deviceptr,success);
    bool blocking_copy = async < 0; // TODO nullptr vs blocking
   
    if ( !success && !if_present ) { 
      LOG_ERROR("gpufortrt_update_device_b: no deviceptr found for deviceptr")
    } else if ( success && blocking_copy ) {
      auto& record = gpufortrt::internal::record_list.records[loc];
      record.copy_to_device(.false.,nullptr);
    } else if ( success ) {
      auto& record = gpufortrt::internal::record_list.records[loc];
      gpufortrt_queue_t queue = gpufortrt::internal::queue_record_list.use_create_queue(async);
      record.copy_to_device(blocking_copy,queue);
    }
  }
}

void gpufortrt_wait(
    int* wait_arg,
    int num_wait,
    int* async,
    int num_async,
    bool condition) {
  if ( condition ) { 
    if ( !gpufortrt::internal::initialized ) {
      throw std::invalid_argument("wait: runtime has not been initialized");
    }
    bool async_specified = async[0] != gpufortrt_async_noval;
    if ( async_specified ) {
      ::wait_async(wait_arg,num_wait,async,num_async);
    } else {
      ::wait(wait_arg,num_wait);
    }
  }
}                
