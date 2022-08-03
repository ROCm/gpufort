// SPDX-License-Identifier: MIT                                                
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "gpufortrt_core.h"

#include <iostream>

#include "assert.h"

#include <hip/hip_runtime_api>

bool gpufortrt::internal::record_list_t::is_initialized() {
  return this->records.capacity() >= gpufortrt::INITIAL_RECORDS_CAPACITY;
}

void gpufortrt::internal::record_list_t::initialize() {
  this->records.reserve(INITIAL_RECORDS_CAPACITY); 
}

void gpufortrt::internal::record_list_t::destroy() {
  for (size_t i = 0; i < records.size(); i++) {
    if ( this->records[i].is_initialized() ) {
      this->records[i].destroy();
    }
  }
  this->records.clear();
}

size_t gpufortrt::internal::record_list_t::find_record(void* hostptr,bool& success) {
  success = false;
  size_t loc = -1;
  size_t offset_bytes = 0;
  if ( hostptr == nullptr ) {
    throw std::invalid_argument("find_record: argument 'hostptr' is null");
  } else {
    for ( size_t i = 0; i < this->records.size(); i++ ) {
      if ( this->records[i].is_subarray(hostptr,0,offset_bytes) ) {
        loc = i
        success = true;
        break;
      }
    }
    // log
    if ( gpufortrt::LOG_LEVEL > 2 ) {
      if ( success ) {
        LOG_INFO(3,"lookup record: " << this->records[loc])
      } else {
        LOG_INFO(3,"lookup record: NOT FOUND")
      }
    }
    return loc;
  }
}

size_t gpufortrt::internal::record_list_t::find_available_record(size_t num_bytes,bool& reuse_existing) {
  size_t loc = this->records.size(); // element after last record
  reuse_existing = false;
  for ( size_t i = 0; i < this->records.size(); i++ ) {
    gpufortrt::internal::record_t& record = record;
    if ( !record.is_initialized() ) {
      loc = i;
      break; // break outer loop
    } else if ( record.is_released() ) {
      loc = i;
      // deallocate memory blocks of regions that have not been reused
      // multiple times in a row
      if ( record.num_bytes < num_bytes
           || num_bytes < record.num_bytes * REUSE_THRESHOLD ) {
        record.dec_refs(gpufortrt::Structured); // decrement structured references
        if ( record.can_be_destroyed(gpufortrt::NUM_REFS_TO_DEALLOCATE) ) {
          record.destroy();
          this->total_memory_bytes -= record.num_bytes;
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

size_t gpufortrt::internal::record_list_t::use_increment_record(
   void* hostptr,
   size_t num_bytes,
   gpufortrt_map_kind_t map_kind,
   counter_t ctr_to_update
   bool blocking_copy,
   gpufortrt_queue_t queue,
   bool never_deallocate) {
  bool success;
  size_t loc = this->find_record(hostptr,success/*inout*/);
  if ( success ) {
    this->record[loc].inc_refs(ctr_to_update);
  } else {
    loc = this->find_available_record(num_bytes,reuse_existing/*inout*/);
    assert(loc >= 0);
    if ( loc == this->records.size() ) {
      gpufortrt::internal::record_t record;
      record.setup(
        gpufortrt::num_records++, // increment global not thread-safe
        hostptr,
        num_bytes,
        map_kind,
        blocking_copy,
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
        this->total_memory_bytes += record.num_bytes;
        LOG_INFO(2,"create record: " << record)
      } else {
        LOG_INFO(2,"reuse record: " << record)
      }
      this->records.push_back(record);
    }
    assert(loc <= this->records.size());
  }
  return loc;
}

void gpufortrt::internal::record_list_t::decrement_release_record(
    void* hostptr,
    gpufortrt_counter_t ctr_to_update,
    bool blocking_copy,
    gpufortrt_queue_t queue,
    bool finalize) {
  bool success;
  size_t loc = this->find_record(hostptr,success/*inout*/);
  if ( success ) {
    assert(loc >= 0);
    assert(loc <= this->records.size());
    gpufortrt::internal::record_t& record = this->records[loc];

    record.dec_refs(ctr_to_update);
    if ( record.hostptr != hostptr ) {
      throw std::invalid_argument(
        "decrement_release_record_: argument 'hostptr' points to section of mapped data");
    } else {
      record.dec_refs(ctr_to_update);
      if ( record.can_be_destroyed(0) || finalize ) {
        // if both structured and dynamic reference counters are zero, 
        // a copyout action is performed
        if ( !finalize &&
             (record.map_kind == gpufortrt_map_kind_copyout
             || record.map_kind == gpufortrt_map_kind_copy ) {
          record.copy_to_host(blocking_copy,queue);
        }
        record.release();
      }
    }
  }
}