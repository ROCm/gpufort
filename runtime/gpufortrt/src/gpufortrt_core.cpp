// SPDX-License-Identifier: MIT
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "gpufortrt_core.h"

#include "gpufort_internal.h"

#include <string>

// global parameters, influenced by environment variables
int gpufortrt::LOG_LEVEL = 0;
size_t gpufortrt::MAX_QUEUES = 64;
size_t gpufortrt::INITIAL_RECORDS_CAPACITY = 4096;
int gpufortrt::BLOCK_SIZE = 32;
double gpufortrt::REUSE_THRESHOLD = 0.9;
int gpufortrt::NUM_REFS_TO_DEALLOCATE = -5;

// global variables
gpufortrt::initialized = 0;
gpufortrt::num_records = 0;
gpufortrt::record_list_t gpufortrt::record_list;
gpufortrt::queue_t       gpufortrt::queues;

void gpufortrt::init() {
  if ( gpufortrt::initialized ) {
    throw std::invalid_argument("init: runtime already initialized");
  else {
    char* val;
    val = getenv("GPUFORTRT_LOG_LEVEL");
    if ( val != nullptr ) {
      GPUFORTRT_LOG_LEVEL = std::stoi(val);
    }
    val = getenv("GPUFORTRT_MAX_QUEUES");
    if ( val != nullptr ) {
      GPUFORTRT_MAX_QUEUES = std::stoi(val);
    }
    val = getenv("GPUFORTRT_MAX_QUEUES");
    if ( val != nullptr ) {
      GPUFORTRT_LOG_LEVEL = std::stoi(val);
    }
    call get_environment_variable("GPUFORTRT_INITIAL_RECORDS_CAPACITY", tmp)
    if (len_trim(tmp) > 0) read(tmp,*) INITIAL_RECORDS_CAPACITY
    !
    call get_environment_variable("GPUFORTRT_BLOCK_SIZE", tmp)
    if (len_trim(tmp) > 0) read(tmp,*) BLOCK_SIZE
    call get_environment_variable("GPUFORTRT_REUSE_THRESHOLD", tmp)
    if (len_trim(tmp) > 0) read(tmp,*) REUSE_THRESHOLD
    call get_environment_variable("GPUFORTRT_NUM_REFS_TO_DEALLOCATE", tmp)
    if (len_trim(tmp) > 0) read(tmp,*) NUM_REFS_TO_DEALLOCATE
    if ( LOG_LEVEL > 0 ) then
      write(*,*) "GPUFORTRT_LOG_LEVEL=",LOG_LEVEL
      write(*,*) "GPUFORTRT_MAX_QUEUES=",MAX_QUEUES
      write(*,*) "GPUFORTRT_INITIAL_RECORDS_CAPACITY=",INITIAL_RECORDS_CAPACITY
      write(*,*) "GPUFORTRT_BLOCK_SIZE=",BLOCK_SIZE
      write(*,*) "GPUFORTRT_REUSE_THRESHOLD=",REUSE_THRESHOLD
      write(*,*) "GPUFORTRT_NUM_REFS_TO_DEALLOCATE=",NUM_REFS_TO_DEALLOCATE
    endif
    !
    call record_list_initialize_()
    allocate(queues_(1:MAX_QUEUES))
    initialized_ = .true.
  }
}

void gpufortrt::shutdown() {
}

void* gpufortrt::use_device_b(void* hostptr,size_t num_bytes,
                              bool condition,bool if_present) {
  
}

void* gpufortrt::present_b(void* hostptr,size_t num_bytes
                           gpufortrt::counter_t ctr_to_update) {
}

void  gpufortrt::dec_struct_refs_b(void* hostptr,int async) {
}

void* gpufortrt::no_create_b(void* hostptr) {
}

void* gpufortrt::create_b(void* hostptr,size_t num_bytes,int async,bool declared_module_var,
                          gpufortrt::counter_t ctr_to_update) {
}

void  gpufortrt::delete_b(void* hostptr,finalize) {
}

void* gpufortrt::copyin_b(void* hostptr,size_t num_bytes,int async,bool declared_module_var,
                          gpufortrt::counter_t ctr_to_update) {
}

void* gpufortrt::copyout_b(void* hostptr,size_t num_bytes,int async,
                           gpufortrt::counter_t ctr_to_update) {
}

void* gpufortrt::copy_b(void* hostptr,size_t num_bytes,int async,
                        gpufortrt::counter_t ctr_to_update) {
  if ( ! initialized_ ) gpufortrt_init();
       if ( hostptr == nullptr) then
 #ifndef NULLPTR_MEANS_NOOP
         throw std::invalid_argument("gpufortrt_copy_b: hostptr must not be null");
 #endif
         deviceptr = c_null_ptr
       else
         loc       = record_list_use_increment_record_(hostptr,num_bytes,&
                       gpufortrt_map_kind_copy,&
                       async,.false.,&
                       update_struct_refs,.false.)
         deviceptr = record_list_%records(loc)%deviceptr
       endif
}

void  gpufortrt::update_host_b(void* hostptr,bool condition,bool if_present,int async) {
}

void  gpufortrt::update_device_b(void* hostptr,bool condition,bool if_present,int async) {
}
