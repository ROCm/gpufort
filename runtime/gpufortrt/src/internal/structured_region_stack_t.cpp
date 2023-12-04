// SPDX-License-Identifier: MIT                                                
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "gpufortrt_core.h"

#include "auxiliary.h"

// structured_region_stack_entry_t
namespace {
  bool is_no_create_entry_without_present_record(const gpufortrt::internal::structured_region_stack_entry_t& entry) {
    return    entry.map_kind  == gpufortrt_map_kind_no_create
           && entry.record == nullptr;
  }
}

void gpufortrt::internal::structured_region_stack_entry_t::to_string(std::ostream& os) const {
  os << "region:"      << this->region_id        
     << ", hostptr:"   << this->hostptr
     << ", num_bytes:" << this->num_bytes
     << ", map_kind:"  << this->map_kind
     << ", recordptr:" << this->record;
}

std::ostream& operator<<(std::ostream& os,const gpufortrt::internal::structured_region_stack_entry_t& entry) {
  entry.to_string(os); 
  return os;
}

gpufortrt::internal::structured_region_stack_entry_t::structured_region_stack_entry_t(
    int region_id,
    gpufortrt_map_kind_t map_kind,
    void* hostptr,
    std::size_t num_bytes) {
  this->region_id = region_id;
  this->map_kind = map_kind;
  this->hostptr = hostptr;
  this->num_bytes = num_bytes;
  this->record = nullptr;
}

gpufortrt::internal::structured_region_stack_entry_t::structured_region_stack_entry_t(
    int region_id,
    gpufortrt_map_kind_t map_kind,
    gpufortrt::internal::record_t* record) {
  this->region_id = region_id;
  this->map_kind = map_kind;
  this->hostptr = record->hostptr;
  this->num_bytes = record->num_bytes;
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

void gpufortrt::internal::structured_region_stack_t::push_back(
        gpufortrt_map_kind_t map_kind,
        void* hostptr,
        std::size_t num_bytes) {
  structured_region_stack_entry_t entry(this->current_region,
                                        map_kind,hostptr,num_bytes);
  LOG_INFO(5,"structured region stack: push entry to back: "<<entry)
  LOG_INFO(6,"  record: "<<*entry.record)
  this->entries.push_back(entry); 
}

std::tuple<bool,gpufortrt::internal::record_t*,std::ptrdiff_t,bool> gpufortrt::internal::structured_region_stack_t::find_record(void* hostptr) const {
  bool success = false;
  gpufortrt::internal::record_t* record = nullptr;
  std::ptrdiff_t offset = -1;
  bool no_create_without_present_record = false;
  //
  for (int i = this->entries.size()-1; i >= 0; i--) {
    auto& entry = this->entries[i];
    auto tup/*is_element,offset*/ = gpufortrt::internal::is_host_data_element(entry,hostptr);
    if ( std::get<0>(tup) ) {
      success = true;
      if ( ::is_no_create_entry_without_present_record(entry) ) {
        no_create_without_present_record = true;
        break;
      } else {
        offset = std::get<1>(tup);
        record = entry.record;
        break;
      }
    }
  }
  return std::make_tuple(success,record,offset,no_create_without_present_record);
}

void gpufortrt::internal::structured_region_stack_t::leave_structured_region(bool blocking,gpufortrt_queue_t queue) {
  LOG_INFO(5,"leave structured region "<<this->current_region)
  for (int i = this->entries.size()-1; i >= 0; i--) {
    auto& entry = this->entries[i];
    if ( entry.region_id == this->current_region ) {
      LOG_INFO(5,"leave structured region: remove stack entry "<<i<<"; "<<entry)
      if ( !::is_no_create_entry_without_present_record(entry) ) {
        entry.record->decrement_release(
                gpufortrt_counter_structured,
                gpufortrt::internal::implies_copy_to_host(entry.map_kind),
                false/*finalize*/,
                blocking,queue);
      }
      this->entries.pop_back();
    } else {
      break;
    }
  }
  this->current_region--;
}
