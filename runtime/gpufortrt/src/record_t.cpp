// SPDX-License-Identifier: MIT                                                
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "gpufortrt_core.h"

#include <iostream>

#include <hip/hip_runtime_api>

std::ostream& operator<<(std::ostream& os, gpufortrt_map_kind_t map_kind);
{
    switch(ce)
    {
       case gpufortrt_map_kind_dec_struct_refs: os << "dec_struct_refs"; break;
       case gpufortrt_map_kind_undefined      : os << "undefined"; break;
       case gpufortrt_map_kind_present        : os << "present"; break;
       case gpufortrt_map_kind_delete         : os << "delete"; break;
       case gpufortrt_map_kind_create         : os << "create"; break;
       case gpufortrt_map_kind_no_create      : os << "no_create"; break;
       case gpufortrt_map_kind_copyin         : os << "copyin"; break;
       case gpufortrt_map_kind_copyout        : os << "copyout"; break;
       case gpufortrt_map_kind_copy           : os << "copy"; break;
       default: throw invalid_argument("operator<<: invalid value for `map_kind`");
    }
    return os;
}

void gpufortrt::internal::record_t::to_string(std::ostream& os) const {
  os << "global id:"          << this->id        
     << ", hostptr:"          << this->hostptr  
     << ", deviceptr:"        << this->deviceptr
     << ", initialized:"      << this->is_initialized() 
     << ", used:"             << this->is_used() 
     << ", released:"         << this->is_released() 
     << ", num_bytes:"        << this->num_bytes 
     << ", struct_refs:"      << this->struct_refs  
     << ", dyn_refs:"         << this->dyn_refs  
     << ", map_kind:"         << 
     static_cast<gpufortrt:map_kind_t>(map_kind);
}

std::ostream& operator<<(std::ostream& os,const gpufortrt::internal::record_t& record) 
{
  record.to_string(os); 
  return os;
}
  
bool gpufortrt::internal::record_t::is_initialized() const {
  return this->deviceptr != nullptr;
}
bool gpufortrt::internal::record_t::is_used() const {
  return this->is_initialized() &&
      this->struct_refs > 0 ||
      this->dyn_refs > 0;
}
bool gpufortrt::internal::record_t::is_released() const {
  return this->is_initialized() &&
      this->struct_refs <= 0 &&
      this->dyn_refs == 0;
}
bool gpufortrt::internal::record_t::can_be_destroyed(int struct_ref_threshold = 0) {
  return this->is_released() && 
      this->struct_refs <= struct_ref_threshold;
}
void gpufortrt::internal::record_t::inc_refs(gpufortrt_counter_t ctr) {
  switch(ctr) {
    case gpufortrt_counter_structured: this->struct_refs++; break;;
    case gpufortrt_counter_dynamic:    this->dyn_refs++; break;;
    case gpufortrt_counter_none: /* do nothing */;break;;
    default: throw invalid_argument("inc_refs: invalid value for 'ctr'");
  }
}

void gpufortrt::internal::record_t::dec_refs(gpufortrt_counter_t ctr) {
  switch(ctr) {
    case gpufortrt_counter_structured: this->struct_refs--; break;;
    case gpufortrt_counter_dynamic:    this->dyn_refs--; break;;
    case gpufortrt_counter_none: /* do nothing */;break;;
    default: throw invalid_argument("dec_refs: invalid value for 'ctr'");
  }
}

void gpufortrt::internal::record_t::release() {
  LOG_INFO(2,"release record:" << *this)
  this->hostptr     = nullptr;
  this->struct_refs = 0;
  this->dyn_refs    = 0;
}

void gpufortrt::internal::record_t::setup(
    int id,
    void* hostptr,
    size_t num_bytes,
    gpufortrt_map_kind_t map_kind,
    bool blocking_copy,
    gpufortrt::internal::queue_t queue,
    bool reuse_existing) {
  this->hostptr = hostptr;
  this->struct_refs = 0;
  this->dyn_refs = 0;
  this->map_kind = map_kind;
  this->num_bytes_used = num_bytes;
  if ( !reuse_existing ) {
    this->id = id; // TODO not thread-safe
    this->num_bytes = gpufortrt::internal::blocked_size(
            num_bytes,gpufortrt::internal::BLOCK_SIZE);
  }
  switch (map_kind) {
    case gpufortrt_map_kind_create:
    case gpufortrt_map_kind_copyout:
      HIP_CHECK(hipMalloc(this->deviceptr,this->num_bytes)) // TODO backend-specific, externalize
      break;
    case gpufortrt_map_kind_copyin:
    case gpufortrt_map_kind_copy:
      HIP_CHECK(hipMalloc(this->deviceptr,this->num_bytes)) // TODO backend-specific, externalize
      this->copy_to_device(blocking_copy,queue);
      break;
  }
}

void gpufortrt::internal::record_t::destroy() {
  // TODO move into C binding
  LOG_INFO(2,"destroy record:" << *this)
  switch (map_kind) {
    case gpufortrt_map_kind_create:
    case gpufortrt_map_kind_copyout:
    case gpufortrt_map_kind_copyin:
    case gpufortrt_map_kind_copy:
      HIP_CHECK(hipFree(this->deviceptr)) // TODO backend-specific, externalize
      break;
  }
  this->hostptr     = nullptr;
  this->struct_refs = 0;
}

void gpufortrt::internal::record_t::copy_to_device(
  bool blocking_copy,
  gpufortrt::internal::queue_t queue) {
  #ifndef BLOCKING_COPIES
  if ( blocking_copy ) then
    // TODO backend-specific, externalize
    HIP_CHECK(hipMemcpyAsync(this->deviceptr,this->hostptr,&
      this->num_bytes_used,hipMemcpyHostToDevice,queue))
  else {
  #endif
    // TODO backend-specific, externalize
    HIP_CHECK(hipMemcpy(this->deviceptr,this->hostptr,this->num_bytes_used,hipMemcpyHostToDevice))
  #ifndef BLOCKING_COPIES
  }
  #endif
}

void gpufortrt::internal::record_t::copy_to_host(
  bool blocking_copy,
  gpufortrt::internal::queue_t queue) {
  #ifndef BLOCKING_COPIES
  if ( blocking_copy ) then
    // TODO backend-specific, externalize
    HIP_CHECK(hipMemcpyAsync(this->hostptr,this->deviceptr,&
      this->num_bytes_used,hipMemcpyDeviceToHost,queue))
  else {
  #endif
    // TODO backend-specific, externalize
    HIP_CHECK(hipMemcpy(this->hostptr,this->deviceptr,this->num_bytes_used,hipMemcpyDeviceToHost))
  #ifndef BLOCKING_COPIES
  }
  #endif
}

bool gpufortrt::internal::record_t::is_subarray(
    void* hostptr, size_t num_bytes, size_t& offset_bytes) const {
  offset_bytes = hostptr - this->hostptr;
  return (offset_bytes >= 0) && ((offset_bytes+num_bytes) < this->num_bytes);    
}
