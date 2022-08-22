// SPDX-License-Identifier: MIT                                                
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "gpufortrt_core.h"

#include <iostream>
#include <sstream>

#include "assert.h"

#include "hip/hip_runtime_api.h"

#include "auxiliary.h"

gpufortrt::internal::record_t& gpufortrt::internal::record_list_t::operator[](const int i) {
  return this->records[i];  
}

const gpufortrt::internal::record_t& gpufortrt::internal::record_list_t::operator[](const int i) const {
  return this->records[i];  
}

void gpufortrt::internal::record_list_t::reserve(int capacity) {
  this->records.reserve(capacity); 
}

void gpufortrt::internal::record_list_t::destroy() {
  for (size_t i = 0; i < records.size(); i++) {
    if ( this->records[i].is_initialized() ) {
      this->records[i].destroy();
    }
  }
  this->records.clear();
}

size_t gpufortrt::internal::record_list_t::find_record(void* hostptr,bool& success) const {
  success = false;
  size_t loc = 0; // typically size_t is unsigned
  if ( hostptr == nullptr ) {
    throw std::invalid_argument("find_record: argument 'hostptr' is null");
  } else {
    for ( size_t i = 0; i < this->records.size(); i++ ) {
      if ( this->records[i].hostptr == hostptr ) {
        loc = i;
        success = true;
        break;
      }
    }
    // log
    if ( gpufortrt::internal::LOG_LEVEL > 2 ) {
      if ( success ) {
        LOG_INFO(3,"record found for hostptr="<<hostptr << "; " << this->records[loc])
      } else {
        LOG_INFO(3,"no record found for hostptr="<<hostptr)
      }
    }
    return loc;
  }
}

size_t gpufortrt::internal::record_list_t::find_record(void* hostptr,size_t num_bytes,bool& success) const {
  success = false;
  size_t loc = 0; // typically size_t is unsigned
  if ( hostptr == nullptr ) {
    throw std::invalid_argument("find_record: argument 'hostptr' is null");
  } else {
    for ( size_t i = 0; i < this->records.size(); i++ ) {
      size_t offset_bytes = 0;
      if ( this->records[i].is_host_data_subset(hostptr,num_bytes,offset_bytes/*inout*/) ) {
        loc = i;
        success = true;
        break;
      }
    }
    // log
    if ( gpufortrt::internal::LOG_LEVEL > 2 ) {
      if ( success ) {
        LOG_INFO(3,"record found for hostptr="<<hostptr << "; " << this->records[loc])
      } else {
        LOG_INFO(3,"no record found for hostptr="<<hostptr)
      }
    }
    return loc;
  }
}

size_t gpufortrt::internal::record_list_t::find_available_record(size_t num_bytes,bool& reuse_existing) {
  size_t loc = this->records.size(); // element after last record
  reuse_existing = false;
  for ( size_t i = 0; i < this->records.size(); i++ ) {
    gpufortrt::internal::record_t& record = this->records[i];
    if ( !record.is_initialized() ) {
      loc = i;
      break; // break outer loop
    } else if ( record.is_released() ) {
      loc = i;
      // deallocate memory blocks of regions that have not been reused
      // multiple times in a row
      if ( record.reserved_bytes < num_bytes
           || num_bytes < record.reserved_bytes * REUSE_THRESHOLD ) {
        record.dec_refs(gpufortrt_counter_structured); // decrement structured references
        if ( record.can_be_destroyed(gpufortrt::internal::NUM_REFS_TO_DEALLOCATE) ) {
          record.destroy();
          this->total_memory_bytes -= record.reserved_bytes;
          loc = i;
          break; // break outer loop
        }
        // continue search
      } else { // existing buffer is appropriately sized
        loc = i;
        reuse_existing = true;
        break; // break outer loop
      }
    }
  }
  return loc;
}

namespace {
  bool verify_is_proper_section_of_host_data(
     gpufortrt::internal::record_t& record,
     void* hostptr,
     size_t num_bytes,
     bool check_restrictions = true) {
    size_t offset_bytes;
    if ( record.is_host_data_subset(hostptr,num_bytes,offset_bytes/*inout*/) ) {
      return true;
    } else if ( check_restrictions ) {
      std::stringstream ss;
      ss << "host data to map (" << hostptr << " x " << num_bytes << " B) is no subset of already existing record's host data ("
         << record.hostptr << " x " << record.used_bytes << " B)";
      throw std::invalid_argument(ss.str());
    }
    return false;
  }
} // namespace

// implements the data clause restriction
size_t gpufortrt::internal::record_list_t::increment_record_if_present(
   gpufortrt_counter_t ctr_to_update,
   void* hostptr,
   size_t num_bytes,
   bool check_restrictions,
   bool& success) {
  success = false;
  bool host_data_contains_hostptr = false;
  size_t loc = this->find_record(hostptr,1,host_data_contains_hostptr/*inout*/);
  if ( host_data_contains_hostptr ) {
    auto& record = this->records[loc];
    if ( ::verify_is_proper_section_of_host_data(record,hostptr,num_bytes,check_restrictions) ) {
      record.inc_refs(ctr_to_update);
      success = true;
    }
  }
  return loc;
}

size_t gpufortrt::internal::record_list_t::create_increment_record(
   gpufortrt_counter_t ctr_to_update,
   void* hostptr,
   size_t num_bytes,
   bool never_deallocate,
   bool allocate_device_buffer,
   bool copy_to_device,
   bool blocking,
   gpufortrt_queue_t queue) {
  bool found = false;
  size_t loc = this->increment_record_if_present(
          ctr_to_update,hostptr,num_bytes,true/*check ...*/,found/*inout*/);
  if ( !found ) { 
    bool reuse_existing = false;
    loc = this->find_available_record(num_bytes,reuse_existing/*inout*/);
    assert(loc >= 0);
    if ( loc == this->records.size() ) {
      gpufortrt::internal::record_t record;
      record.setup(
        gpufortrt::internal::num_records++, // increment global not thread-safe
        hostptr,
        num_bytes,
        allocate_device_buffer,
        copy_to_device,
        blocking,
        queue,
        reuse_existing/*in*/);
      // Initialize fixed-size Fortran module variables with 
      // structured references counter set to '1'.
      if ( ctr_to_update == gpufortrt_counter_structured
           && never_deallocate ) {
         record.struct_refs = 1; 
      }
      // Update consumed memory statistics only if
      // no record was reused, i.e. a new one was created
      if ( !reuse_existing ) {
        this->total_memory_bytes += record.reserved_bytes;
        LOG_INFO(3,"create record: " << record)
      } else {
        LOG_INFO(3,"reuse record: " << record)
      }
      record.inc_refs(ctr_to_update);
      this->records.push_back(record);
    }
    assert(loc <= this->records.size());
  }
  return loc;
}

void gpufortrt::internal::record_list_t::decrement_release_record(
  gpufortrt_counter_t ctr_to_update,
  void* hostptr,
  size_t num_bytes,
  bool copyout,
  bool finalize,
  bool blocking,
  gpufortrt_queue_t queue) {
  bool host_data_contains_hostptr;
  size_t loc = this->find_record(hostptr,1,host_data_contains_hostptr/*inout*/);
  if ( host_data_contains_hostptr ) {
    assert(loc >= 0);
    assert(loc <= this->records.size());
    gpufortrt::internal::record_t& record = this->records[loc];
    if ( ::verify_is_proper_section_of_host_data(record,hostptr,num_bytes) ) { 
      record.decrement_release(
        ctr_to_update,
        copyout,finalize,
        blocking,queue);
    }
  }
}
