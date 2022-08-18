// SPDX-License-Identifier: MIT                                                
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "gpufortrt_core.h"

#include "auxiliary.h"

// structured_region_stack_entry_t

void gpufortrt::internal::structured_region_stack_entry_t::to_string(std::ostream& os) const {
  os << "region id:"          << this->region_id        
     << ", recordptr:"        << this->record;
}

std::ostream& operator<<(std::ostream& os,const gpufortrt::internal::structured_region_stack_entry_t& entry) {
  entry.to_string(os); 
  return os;
}

gpufortrt::internal::structured_region_stack_entry_t::structured_region_stack_entry_t(
  int region_id,gpufortrt::internal::record_t& record) {
  this->region_id = region_id;
  this->record    = &record;
}

// structured_region_stack_t 

void gpufortrt::internal::structured_region_stack_t::reserve(int capacity) {
  this->entries.reserve(capacity);
}

void gpufortrt::internal::structured_region_stack_t::enter_structured_region() {
  this->current_region++;
  LOG_INFO(5,"enter structured region "<<this->current_region)
}

void gpufortrt::internal::structured_region_stack_t::push_back(gpufortrt::internal::record_t& record) {
  structured_region_stack_entry_t entry(this->current_region,record);
  LOG_INFO(5,"structured region stack: push entry to back: "<<entry)
  LOG_INFO(6,"  record: "<<*entry.record)
  this->entries.push_back(entry); 
}
      
gpufortrt::internal::record_t* gpufortrt::internal::structured_region_stack_t::find_in_current_region(void* hostptr,size_t num_bytes) {
  for (int i = this->entries.size()-1; i >= 0; i--) {
    auto& entry = this->entries[i];
    if ( entry.region_id == this->current_region ) {
      size_t offset_bytes = 0;
      if ( entry.record->is_subarray(hostptr,num_bytes,offset_bytes/*inout*/) ) {
         return entry.record;
      }
    } else {
      return nullptr;
    }
  }
  return nullptr;
}

gpufortrt::internal::record_t* gpufortrt::internal::structured_region_stack_t::find(void* hostptr,size_t num_bytes) {
  for (int i = this->entries.size()-1; i >= 0; i--) {
    auto& entry = this->entries[i];
    size_t offset_bytes;
    if ( entry.record->is_subarray(hostptr,num_bytes,offset_bytes/*inout*/) ) {
      return entry.record;
    } 
  }
  return nullptr;
}

void gpufortrt::internal::structured_region_stack_t::leave_structured_region(bool blocking,gpufortrt_queue_t queue) {
  LOG_INFO(5,"leave structured region "<<this->current_region)
  for (int i = this->entries.size()-1; i >= 0; i--) {
    auto& entry = this->entries[i];
    if ( entry.region_id == this->current_region ) {
      LOG_INFO(5,"leave structured region: remove stack entry "<<i<<"; "<<entry)
      entry.record->structured_decrement_release(blocking,queue);
      this->entries.pop_back();
    } else {
      break;
    }
  }
  this->current_region--;
}