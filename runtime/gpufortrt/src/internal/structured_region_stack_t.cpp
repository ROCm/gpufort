// SPDX-License-Identifier: MIT                                                
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "gpufortrt_core.h"

// structured_region_stack_entry_t

void gpufortrt::internal::structured_region_stack_entry_t::structured_region_stack_entry_t(
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
}

void gpufortrt::internal::structured_region_stack_t::push_record(gpufortrt::internal::record_t& record) {
  structured_region_stack_entry_t entry(this->current_region,&record);
  this->entries.push_back(entry); 
}
      
gpufortrt::internal::record_t* gpufortrt::internal::structured_region_stack_t::find_in_current_region(void* hostptr) {
  for (int i = this->entries.size(); i >= 0; i--) {
    auto& entry = this->entries[i];
    size_t offset_bytes;
    if ( entry.region_id == this->current_region ) {
      if ( entry.record->is_subarray(hostptr,0,offset_bytes/*inout*/) ) {
         return entry.record;
      }
    } else {
      return nullptr;
    }
  }
}

gpufortrt::internal::record_t* gpufortrt::internal::structured_region_stack_t::find(void* hostptr) {
  for (int i = this->entries.size(); i >= 0; i--) {
    auto& entry = this->entries[i];
    size_t offset_bytes;
    if ( entry.record->is_subarray(hostptr,0,offset_bytes/*inout*/) ) {
       return entry.record;
    } 
  }
  return nullptr;
}

void gpufortrt::internal::structured_region_stack_t::leave_structured_region(bool blocking,gpufortrt_queue_t queue) {
  for (int i = this->entries.size(); i >= 0; i--) {
    auto& entry = this->entries[i];
    if ( entry.region_id == this->current_region ) {
      entry.decrement_release_record(
        gpufortrt_counter_structured,
        blocking,queue,
        false/*finalize*/) {
      this->entries.erase(i);
    } else {
      break;
    }
  }
  this->current_region--;
}
