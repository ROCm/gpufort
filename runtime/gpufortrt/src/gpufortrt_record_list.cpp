// SPDX-License-Identifier: MIT                                                
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "gpufortrt_core.h"

#include <iostream>

#include "assert.h"

#include <hip/hip_runtime_api>

bool gpufortrt::record_list::is_initialized() {
  return this->records.capacity() >= gpufortrt::INITIAL_RECORDS_CAPACITY;
}

void gpufortrt::record_list::initialize() {
  this->records.reserve(INITIAL_RECORDS_CAPACITY); 
}

void gpufortrt::record_list::grow() {
  this->records.reserve(2*records.capacity());
}

void gpufortrt::record_list::destroy() {
  for (size_t i = 0; i < records.size(); i++) {
    if ( this->records[i].is_initialized() ) {
      this->records[i].destroy();
    }
  }
  this->records.clear();
}

size_t gpufortrt::record_list::find_record(void* hostptr,bool& success) {
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
        LOG_ERROR(3,"lookup record: " << this->records[loc])
      } else {
        LOG_ERROR(3,"lookup record: NOT FOUND")
      }
    }
    return loc;
  }
}

size_t gpufortrt::record_list::find_available_record(size_t num_bytes,bool& reuse_existing) {
  size_t loc = this->records.size(); // element after last record
  reuse_existing = false;
  for ( size_t i = 0; i < this->records.size(); i++ ) {
    gpufortrt::record_t& record = record;
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

size_t gpufortrt::record_list::use_increment_record(
   void* hostptr,
   size_t num_bytes,
   gpufortrt::map_kind_t map_kind,
   counter_t ctr_to_update
   bool blocking_copy,
   gpufortrt::queue_t queue,
   bool declared_module_var) {
  bool success;
  size_t loc = this->find_record(hostptr,success/*inout*/);
  if ( success ) {
    this->record[loc].inc_refs(ctr_to_update);
  } else {
    loc = this->find_available_record(num_bytes,reuse_existing/*inout*/);
    if ( loc >= this->records.capacity() ) {
      this->grow();
    }
    if ( loc >= this->records.size() ) {
      this->records.emplace_back(); // initialize record at vector's end
    }
    assert(loc >= 0);
    assert(loc <= this->records.size());
    gpufortrt::record_t& record = this->records[loc];
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
    if ( ctr_to_update == gpufortrt::counter_t::Structured
         && declared_module_var ) {
       record.struct_refs = 1; 
    }
    // Update consumed memory statistics only if
    // no record was reused, i.e. a new one was created
    if ( !reuse_existing ) {
      this->total_memory_bytes += record.num_bytes;
    }
    if ( gpufortrt::LOG_LEVEL > 1 ) {
      if ( reuse_existing ) {
        LOG_ERROR(2,"reuse record: " << record)
      } else {
        LOG_ERROR(2,"create record: " << record)
      }
    }
  }
  return loc;
}

void gpufortrt::record_list::decrement_release_record(
  void* hostptr,
  gpufortrt::counter_t ctr_to_update,
  bool veto_copy_to_host,
  bool blocking_copy,
  gpufortrt::queue_t queue) {
  bool success;
  size_t loc = this->find_record(hostptr,success/*inout*/);
  if ( success ) {
    assert(loc >= 0);
    assert(loc <= this->records.size());
    gpufortrt::record_t& record = this->records[loc];

    record.dec_refs(ctr_to_update);
    if ( record.hostptr != hostptr ) {
      throw std::invalid_argument(
              "decrement_release_record_: argument 'hostptr' points to section of mapped data");
    } else {
      record.dec_refs(ctr_to_update);
      if ( record.can_be_destroyed(0) ) {
        // if both structured and dynamic reference counters are zero, 
        // a copyout action is performed
        if ( !veto_copy_to_host &&
             (record.map_kind == gpufortrt::map_kind_t::Copyout
             || record.map_kind == gpufortrt::map_kind_t::Copy ) {
          record.copy_to_host(blocking_copy,queue);
        }
      }
    }
  }
}
