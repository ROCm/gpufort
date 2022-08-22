// SPDX-License-Identifier: MIT                                                
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "gpufortrt_core.h"

#include "auxiliary.h"

// structured_region_stack_entry_t

void gpufortrt::internal::structured_region_stack_entry_t::to_string(std::ostream& os) const {
  os << "region:"      << this->region_id        
     << ", map_kind:"    << this->map_kind
     << ", recordptr:" << this->record;
}

std::ostream& operator<<(std::ostream& os,const gpufortrt::internal::structured_region_stack_entry_t& entry) {
  entry.to_string(os); 
  return os;
}

gpufortrt::internal::structured_region_stack_entry_t::structured_region_stack_entry_t(
  int region_id,
  gpufortrt_map_kind_t map_kind,
  gpufortrt::internal::record_t* record) {
  this->region_id = region_id;
  this->map_kind = map_kind;
  this->record = record;
}

// structured_region_stack_t 

void gpufortrt::internal::structured_region_stack_t::reserve(int capacity) {
  this->entries.reserve(capacity);
}

void gpufortrt::internal::structured_region_stack_t::enter_structured_region() {
  this->current_region++;
  LOG_INFO(5,"enter structured region "<<this->current_region)
}

void gpufortrt::internal::structured_region_stack_t::push_back(
        gpufortrt_map_kind_t map_kind,
        gpufortrt::internal::record_t* record) {
  structured_region_stack_entry_t entry(this->current_region,
                                        map_kind,record);
  LOG_INFO(5,"structured region stack: push entry to back: "<<entry)
  LOG_INFO(6,"  record: "<<*entry.record)
  this->entries.push_back(entry); 
}
     
namespace {
  bool is_no_create_entry_without_present_record(const gpufortrt::internal::structured_region_stack_entry_t& entry) {
    return    entry.map_kind  == gpufortrt_map_kind_no_create
           && entry.record == nullptr;
  }
}

gpufortrt::internal::record_t* gpufortrt::internal::structured_region_stack_t::find_record(
        void* hostptr,size_t num_bytes,
        bool& no_create_without_present_record) const {
  no_create_without_present_record = false;
  for (int i = this->entries.size()-1; i >= 0; i--) {
    auto& entry = this->entries[i];
    if ( ::is_no_create_entry_without_present_record(entry) ) {
      no_create_without_present_record = true;
    } else {
      size_t offset_bytes;
      if ( entry.record->is_host_data_subset(hostptr,num_bytes,offset_bytes/*inout*/) ) {
        return entry.record;
      }
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
      if ( !::is_no_create_entry_without_present_record(entry) ) {
        entry.record->structured_decrement_release(
                entry.map_kind,blocking,queue);
      }
      this->entries.pop_back();
    } else {
      break;
    }
  }
  this->current_region--;
}