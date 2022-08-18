// SPDX-License-Identifier: MIT                                                
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "gpufortrt_core.h"

#include <iostream>

#include "hip/hip_runtime_api.h"

#include "auxiliary.h"

void gpufortrt::internal::record_t::to_string(std::ostream& os) const {
  os << "global id:"          << this->id        
     << ", hostptr:"          << this->hostptr  
     << ", deviceptr:"        << this->deviceptr
     << ", initialized:"      << this->is_initialized() 
     << ", used:"             << this->is_used() 
     << ", released:"         << this->is_released() 
     << ", num_bytes_used:"   << this->num_bytes_used 
     << ", num_bytes:"        << this->num_bytes 
     << ", struct_refs:"      << this->struct_refs  
     << ", dyn_refs:"         << this->dyn_refs  
     << ", map_kind:"         << 
     static_cast<gpufortrt_map_kind_t>(this->map_kind);
}

std::ostream& operator<<(std::ostream& os,const gpufortrt::internal::record_t& record) {
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
bool gpufortrt::internal::record_t::can_be_destroyed(int struct_ref_threshold) {
  return this->is_released() && 
      this->struct_refs <= struct_ref_threshold;
}
void gpufortrt::internal::record_t::inc_refs(gpufortrt_counter_t ctr) {
  switch(ctr) {
    case gpufortrt_counter_structured: 
        this->struct_refs++;
        LOG_INFO(4,"increment " << ctr << " references; result: " << *this)
        break;;
    case gpufortrt_counter_dynamic:    
        this->dyn_refs++;
        LOG_INFO(4,"increment " << ctr << " references; result: " << *this)
        break;;
    case gpufortrt_counter_none: 
        /* do nothing */;break;;
    default: throw std::invalid_argument("inc_refs: std::invalid value for 'ctr'");
  }
}

void gpufortrt::internal::record_t::dec_refs(gpufortrt_counter_t ctr) {
  switch(ctr) {
    case gpufortrt_counter_structured: 
        this->struct_refs--;
        LOG_INFO(4,"decrement " << ctr << " references; result: " << *this)
        break;;
    case gpufortrt_counter_dynamic:
        this->dyn_refs--;
        LOG_INFO(4,"decrement " << ctr << " references; result: " << *this)
        break;;
    case gpufortrt_counter_none:
        /* do nothing */;break;;
    default: throw std::invalid_argument("dec_refs: std::invalid value for 'ctr'");
  }
}

void gpufortrt::internal::record_t::release() {
  LOG_INFO(3,"release record; " << *this)
  this->hostptr     = nullptr;
  this->struct_refs = 0;
  this->dyn_refs    = 0;
}

namespace {
  size_t blocked_size(size_t num_bytes,size_t block_size) {
    return (((num_bytes)+block_size-1)/block_size) * block_size;
  }
}

void gpufortrt::internal::record_t::setup(
    int id,
    void* hostptr,
    size_t num_bytes,
    gpufortrt_map_kind_t map_kind,
    bool blocking,
    gpufortrt_queue_t queue,
    bool reuse_existing) {
  this->hostptr = hostptr;
  this->struct_refs = 0;
  this->dyn_refs = 0;
  this->map_kind = map_kind;
  this->num_bytes_used = num_bytes;
  if ( !reuse_existing ) {
    this->id = id; // TODO not thread-safe
    this->num_bytes = ::blocked_size(num_bytes,gpufortrt::internal::BLOCK_SIZE);
  }
  switch (map_kind) {
    case gpufortrt_map_kind_create:
    case gpufortrt_map_kind_copyout:
      HIP_CHECK(hipMalloc(&this->deviceptr,this->num_bytes)) // TODO backend-specific, externalize
      break;
    case gpufortrt_map_kind_copyin:
    case gpufortrt_map_kind_copy:
      HIP_CHECK(hipMalloc(&this->deviceptr,this->num_bytes)) // TODO backend-specific, externalize
      this->copy_to_device(blocking,queue);
      break;
    default:
      // ignore
      break;
  }
}

void gpufortrt::internal::record_t::destroy() {
  // TODO move into C binding
  LOG_INFO(3,"destroy record; " << *this)
  switch (map_kind) {
    case gpufortrt_map_kind_create:
    case gpufortrt_map_kind_copyout:
    case gpufortrt_map_kind_copyin:
    case gpufortrt_map_kind_copy:
      HIP_CHECK(hipFree(this->deviceptr)) // TODO backend-specific, externalize
      break;
    default:
      // ignore
      break;
  }
  this->hostptr     = nullptr;
  this->struct_refs = 0;
}

void gpufortrt::internal::record_t::copy_to_device(
  bool blocking,
  gpufortrt_queue_t queue) {
  #ifndef BLOCKING_COPIES
  if ( !blocking ) {
    // TODO backend-specific, externalize
    HIP_CHECK(hipMemcpyAsync(this->deviceptr,this->hostptr,
      this->num_bytes_used,hipMemcpyHostToDevice,queue))
} else {
  #endif
    // TODO backend-specific, externalize
    HIP_CHECK(hipMemcpy(this->deviceptr,this->hostptr,this->num_bytes_used,hipMemcpyHostToDevice))
  #ifndef BLOCKING_COPIES
  }
  #endif
}

void gpufortrt::internal::record_t::copy_to_host(
  bool blocking,
  gpufortrt_queue_t queue) {
  #ifndef BLOCKING_COPIES
  if ( !blocking ) {
    // TODO backend-specific, externalize
    HIP_CHECK(hipMemcpyAsync(this->hostptr,this->deviceptr,
      this->num_bytes_used,hipMemcpyDeviceToHost,queue))
} else {
  #endif
    // TODO backend-specific, externalize
    HIP_CHECK(hipMemcpy(this->hostptr,this->deviceptr,this->num_bytes_used,hipMemcpyDeviceToHost))
  #ifndef BLOCKING_COPIES
  }
  #endif
}

bool gpufortrt::internal::record_t::is_host_data_subset(
    void* hostptr, size_t num_bytes, size_t& offset_bytes) const {
  if ( num_bytes < 1 ) {
    throw std::invalid_argument("is_host_data_subset: argument `num_bytes` must be greater than or equal to 1");
  }
  offset_bytes = static_cast<char*>(hostptr) - static_cast<char*>(this->hostptr);
  return (offset_bytes >= 0) && ((offset_bytes+num_bytes) <= this->num_bytes_used);    
}

void gpufortrt::internal::record_t::copy_section_to_device(
  void* hostptr,
  size_t num_bytes,
  bool blocking,
  gpufortrt_queue_t queue) {
  size_t offset_bytes;
  if ( !this->is_host_data_subset(hostptr,num_bytes,offset_bytes/*inout*/) ) {
    throw std::invalid_argument("copy_section_to_device: data section is no subset of record data");
  }
  void* deviceptr_section_begin = static_cast<void*>(
      static_cast<char*>(this->deviceptr) + offset_bytes);
  LOG_INFO(3,"copy_section_to_device" 
           << "; hostptr:" << hostptr
           << ", deviceptr:" << deviceptr_section_begin
           << ", num_bytes:" << num_bytes
           << ", offset_bytes:" << offset_bytes
           << ", record_hostptr:" << this->hostptr
           << ", record_deviceptr:" << this->deviceptr
           << ", record_num_bytes_used: " << this->num_bytes_used)
  #ifndef BLOCKING_COPIES
  if ( !blocking ) {
    // TODO backend-specific, externalize
    HIP_CHECK(hipMemcpyAsync(
      deviceptr_section_begin,
      hostptr,
      num_bytes,
      hipMemcpyHostToDevice,queue))
} else {
  #endif
    // TODO backend-specific, externalize
    HIP_CHECK(hipMemcpy(
      deviceptr_section_begin,
      hostptr,
      num_bytes,
      hipMemcpyHostToDevice))
  #ifndef BLOCKING_COPIES
  }
  #endif
}

void gpufortrt::internal::record_t::copy_section_to_host(
  void* hostptr,
  size_t num_bytes,
  bool blocking,
  gpufortrt_queue_t queue) {
  size_t offset_bytes;
  if ( !this->is_host_data_subset(hostptr,num_bytes,offset_bytes/*inout*/) ) {
    throw std::invalid_argument("copy_section_to_host: data section is no subset of record data");
  }
  void* deviceptr_section_begin = static_cast<void*>(
      static_cast<char*>(this->deviceptr) + offset_bytes);
  LOG_INFO(3,"copy_section_to_host" 
           << "; hostptr:" << hostptr
           << ", deviceptr:" << deviceptr_section_begin
           << ", num_bytes:" << num_bytes
           << ", offset_bytes:" << offset_bytes
           << ", record_hostptr:" << this->hostptr
           << ", record_deviceptr:" << this->deviceptr
           << ", record_num_bytes_used: " << this->num_bytes_used)
  #ifndef BLOCKING_COPIES
  if ( !blocking ) {
    // TODO backend-specific, externalize
    HIP_CHECK(hipMemcpyAsync(
      hostptr,
      deviceptr_section_begin,
      num_bytes,
      hipMemcpyDeviceToHost,queue))
} else {
  #endif
    // TODO backend-specific, externalize
    HIP_CHECK(hipMemcpy(
      hostptr,
      deviceptr_section_begin,
      num_bytes,
      hipMemcpyDeviceToHost))
  #ifndef BLOCKING_COPIES
  }
  #endif
}

void gpufortrt::internal::record_t::structured_decrement_release(
     bool blocking, gpufortrt_queue_t queue) {
  this->dec_refs(gpufortrt_counter_structured);
  if ( this->can_be_destroyed(0) ) {
    // if both structured and dynamic reference counters are zero, 
    // a copyout action is performed
    if (  this->map_kind == gpufortrt_map_kind_copyout
       || this->map_kind == gpufortrt_map_kind_copy ) {
      this->copy_to_host(blocking,queue);
    }
    this->release();
  }
}

void gpufortrt::internal::record_t::unstructured_decrement_release(
        bool copyout, bool finalize,
        bool blocking, gpufortrt_queue_t queue) {
  this->dec_refs(gpufortrt_counter_dynamic);
  if ( this->can_be_destroyed(0) || finalize ) {
    // if both structured and dynamic reference counters are zero, 
    // a copyout action is performed
    if (  copyout ) {
      this->copy_to_host(blocking,queue);
    }
    this->release();
  }
}
