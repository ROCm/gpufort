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
gpufortrt::internal::queue_record_list_t gpufortrt::queue_record_list;

void gpufortrt_mapping_init(
    gpufortrt_mapping_t* mapping,
    void* hostptr,
    size_t num_bytes,
    gpufortrt::mapkind_t map_kind,
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
    LOG_INFO(1,"Initialized runtime")
    LOG_INFO(1,"GPUFORTRT_LOG_LEVEL=" << gpufortrt::internal::LOG_LEVEL)
    LOG_INFO(1,"GPUFORTRT_INITIAL_QUEUE_RECORDS_CAPACITY=" << gpufortrt::internal::INITIAL_QUEUE_RECORDS_CAPACITY)
    LOG_INFO(1,"GPUFORTRT_INITIAL_RECORDS_CAPACITY=" << gpufortrt::internal::INITIAL_RECORDS_CAPACITY)
    LOG_INFO(1,"GPUFORTRT_BLOCK_SIZE=" << gpufortrt::internal::BLOCK_SIZE)
    LOG_INFO(1,"GPUFORTRT_REUSE_THRESHOLD=" << gpufortrt::internal::REUSE_THRESHOLD)
    LOG_INFO(1,"GPUFORTRT_NUM_REFS_TO_DEALLOCATE=" << gpufortrt::internal::NUM_REFS_TO_DEALLOCATE)
  }
}

void gpufortrt_shutdown() {
  if ( gpufortrt::internal::initialized ) {
    LOG_ERROR("gpufortrt_shutdown: runtime has not been initialized")
  }
  gpufortrt::record_list.destroy();
  gpufortrt::queue_record_list.destroy();
  LOG_INFO(1,"Shutdown runtime")
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
      gpufortrt::queue_t queue = gpufortrt::queue_record_list.use_create_queue(async);
      
      size_t loc = gpufortrt::internal::record_list.use_increment_record(
        hostptr,
        num_bytes,&
        map_kind,&
        ctr_to_update,
        blocking_copy,
        queue,
        never_deallocate);
      return gpufortrt::internal::record_list.records(loc).deviceptr;
    }
  }
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
        auto& record = gpufortrt::internal::record_list.records(loc);
        bool fits = record.is_subarray(hostptr,num_bytes,offset_bytes); // TODO might not fit, i.e. only subarray
                                                                                          // might have been mapped before
        return static_cast<void*>(static_cast<char*>(record%deviceptr) + offset_bytes);
    } else if ( if_present ) {
        resultptr = hostptr;
    } else {
        print *, "ERROR: did not find record for hostptr:"
        CALL print_cptr(hostptr)
        ERROR STOP "gpufortrt_use_device_b: no record found for hostptr"
    }
  }
}

void* gpufortrt_present_b(void* hostptr,size_t num_bytes
                          gpufortrt_counter_t ctr_to_update) {
  return ::run_use_increment_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_present,
    async,
    true, 
    ctr_to_update);
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
    gpufortrt::queue_t queue = gpufortrt::queue_record_list.use_create_queue(async);
    
    gpufortrt::internal::record_list.decrement_release_record(
      hostptr,
      gpufortrt_counter_t_structured,
      veto_copy_to_host,
      blocking_copy,
      queue,
      false);
  }
}

void* gpufortrt_no_create_b(void* hostptr) {
  return ::run_use_increment_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_no_create,
    async,
    true, 
    ctr_to_update);
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

void* gpufortrt_create_b(void* hostptr,size_t num_bytes,int async,bool never_deallocate,
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

void* gpufortrt_copyout_b(void* hostptr,size_t num_bytes,int async,
                           gpufortrt_counter_t ctr_to_update) {
  return ::run_use_increment_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_copyout,
    async,
    false,
    ctr_to_update);
}

void* gpufortrt_copy_b(void* hostptr,size_t num_bytes,int async,
                       gpufortrt_counter_t ctr_to_update) {
  return ::run_use_increment_action(
    hostptr,
    num_bytes,
    gpufortrt_map_kind_copy,
    async,
    false,
    ctr_to_update);
}

void  gpufortrt_update_host_b(void* hostptr,bool condition,bool if_present,int async) {
}

void  gpufortrt_update_device_b(void* hostptr,bool condition,bool if_present,int async) {
}
