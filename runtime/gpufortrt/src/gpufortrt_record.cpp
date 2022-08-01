// SPDX-License-Identifier: MIT                                                
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "gpufortrt_core.h"

#include <iostream>

#include <hip/hip_runtime_api>

std ostream& operator<<(std::ostream& os, gpufortrt::map_kind_t map_kind);
{
    switch(ce)
    {
       case gpufortrt::map_kind_t::Dec_struct_refs: os << "dec_struct_refs"; break;
       case gpufortrt::map_kind_t::Undefined      : os << "undefined"; break;
       case gpufortrt::map_kind_t::Present        : os << "present"; break;
       case gpufortrt::map_kind_t::Delete         : os << "delete"; break;
       case gpufortrt::map_kind_t::Create         : os << "create"; break;
       case gpufortrt::map_kind_t::No_create      : os << "no_create"; break;
       case gpufortrt::map_kind_t::Copyin         : os << "copyin"; break;
       case gpufortrt::map_kind_t::Copyout        : os << "copyout"; break;
       case gpufortrt::map_kind_t::Copy           : os << "copy"; break;
       default: os.setstate(std ios_base failbit);;
    }
    return os;
}

void gpufortrt::record_t::to_string(std::ostream& os) const {
  os << std::unitbuf
     << "global id:"          << id        
     << ", hostptr:"          << hostptr  
     << ", deviceptr:"        << deviceptr
     << ", initialized:"      << initialized 
     << ", used:"             << used 
     << ", released:"         << released 
     << ", num_bytes:"        << num_bytes 
     << ", struct_refs:"      << struct_refs  
     << ", dyn_refs:"         << dyn_refs  
     << ", map_kind:"         << 
     static_cast<gpufortrt:map_kind_t>(map_kind);
}
  
bool gpufortrt::record_t::is_initialized() const {
  return this->deviceptr != nullptr;
}
bool gpufortrt::record_t::is_used() const {
  return this->is_initialized() &&
      this->struct_refs > 0 ||
      this->dyn_refs > 0;
}
bool gpufortrt::record_t::is_released() const {
  return this->is_initialized() &&
      this->struct_refs <= 0 &&
      this->dyn_refs == 0;
}
bool gpufortrt::record_t::can_be_destroyed(int struct_ref_threshold = 0) {
  return this->is_released() && 
      this->struct_refs <= struct_ref_threshold;
}
void gpufortrt::record_t::inc_refs(gpufortrt::counter_t ctr) {
  switch(ctr) {
    case gpufortrt::counter_t::Structured: this->struct_refs++; break;;
    case gpufortrt::counter_t::Dynamic:    this->dyn_refs++; break;;
    case gpufortrt::counter_t::None: /* do nothing */;break;;
    default: throw invalid_argument("inc_refs: invalid value for 'ctr'");
  }
}

void gpufortrt::record_t::dec_refs(gpufortrt::counter_t ctr) {
  switch(ctr) {
    case gpufortrt::counter_t::Structured: this->struct_refs--; break;;
    case gpufortrt::counter_t::Dynamic:    this->dyn_refs--; break;;
    case gpufortrt::counter_t::None: /* do nothing */;break;;
    default: throw invalid_argument("dec_refs: invalid value for 'ctr'");
  }
}

void gpufortrt::record_t::release() {
  if ( gpufortrt::LOG_LEVEL > 1 ) { // TODO move into C-binding
    LOG_ERROR(2,"release record:" << *this)
  }
  this->hostptr     = nullptr;
  this->struct_refs = 0;
}

void gpufortrt::record_t::setup(
    int id,
    void* hostptr,
    size_t num_bytes,
    gpufortrt::map_kind_t map_kind,
    bool blocking_copy,
    gpufortrt::queue_t queue,
    bool reuse_existing) {
  this->hostptr = hostptr;
  this->struct_refs = 0;
  this->dyn_refs = 0;
  this->map_kind = map_kind;
  this->num_bytes_used = num_bytes;
  if ( !reuse_existing ) {
    this->id = id; // TODO not thread-safe
    this->num_bytes = gpufortrt::blocked_size(num_bytes);
  }
  switch (map_kind) {
    case gpufortrt::map_kind_t::Create:
    case gpufortrt::map_kind_t::Copyout:
      HIP_CHECK(hipMalloc(this->deviceptr,this->num_bytes)) // TODO backend-specific, externalize
      break;
    case gpufortrt::map_kind_t::Copyin:
    case gpufortrt::map_kind_t::Copy:
      HIP_CHECK(hipMalloc(this->deviceptr,this->num_bytes)) // TODO backend-specific, externalize
      this->copy_to_device(blocking_copy,queue);
      break;
  }
}

void gpufortrt::record_t::destroy() {
  if ( gpufortrt::LOG_LEVEL > 1 ) { // TODO move into C-binding
    LOG_ERROR(2,"destroy record:" << *this)
  }
  switch (map_kind) {
    case gpufortrt::map_kind_t::Create:
    case gpufortrt::map_kind_t::Copyout:
    case gpufortrt::map_kind_t::Copyin:
    case gpufortrt::map_kind_t::Copy:
      HIP_CHECK(hipFree(this->deviceptr)) // TODO backend-specific, externalize
      break;
  }
  this->hostptr     = nullptr;
  this->struct_refs = 0;
}

void gpufortrt::record_t::copy_to_device(
  bool blocking_copy,
  gpufortrt::queue_t queue) {
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

void gpufortrt::record_t::copy_to_host(
  bool blocking_copy,
  gpufortrt::queue_t queue) {
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

bool gpufortrt::record_t::is_subarray(
    void* hostptr, size_t num_bytes, size_t& offset_bytes) const {
  offset_bytes = hostptr - this->hostptr;
  return (offset_bytes >= 0) && ((offset_bytes+num_bytes) < this->num_bytes);    
}
