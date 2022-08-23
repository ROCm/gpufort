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
  for (std::size_t i = 0; i < records.size(); i++) {
    if ( this->records[i].is_initialized() ) {
      this->records[i].destroy();
    }
  }
  this->records.clear();
}

std::tuple<bool,std::size_t,std::ptrdiff_t> gpufortrt::internal::record_list_t::find_record(void* hostptr) const {
  bool success = false;
  std::size_t loc = 0; // typically std::size_t is unsigned
  std::ptrdiff_t offset = -1;
  if ( hostptr == nullptr ) {
    throw std::invalid_argument("find_record: argument 'hostptr' is null");
  } else {
    for ( std::size_t i = 0; i < this->records.size(); i++ ) {
      auto& record = this->records[i];
      if ( record.is_initialized() ) {
        auto tup/*is_element,offset*/ = gpufortrt::internal::is_host_data_element(record,hostptr);
        if ( std::get<0>(tup) ) {
          success = true;
          loc = i;
          offset = std::get<1>(tup);
          break;
        }
      }
    }
    // log
    if ( gpufortrt::internal::LOG_LEVEL > 2 ) {
      if ( success) {
        LOG_INFO(3,"record found for hostptr="<<hostptr << "; " << this->records[loc])
      } else {
        LOG_INFO(3,"no record found for hostptr="<<hostptr)
      }
    }
    return std::make_tuple(success,loc,offset);
  }
}

std::tuple<bool,std::size_t,std::ptrdiff_t> gpufortrt::internal::record_list_t::find_record(void* hostptr,std::size_t num_bytes) const {
  bool success = false;
  std::size_t loc = 0; // typically std::size_t is unsigned
  std::ptrdiff_t offset = -1;
  if ( hostptr == nullptr ) {
    throw std::invalid_argument("find_record: argument 'hostptr' is null");
  } else {
    for ( std::size_t i = 0; i < this->records.size(); i++ ) {
      auto& record = this->records[i];
      if ( record.is_initialized() ) {
        auto tup/*is_subset,offset*/ = gpufortrt::internal::is_host_data_subset(record,hostptr,num_bytes);
        if ( std::get<0>(tup) ) {
          success = true;
          loc = i;
          offset = std::get<1>(tup);
          break;
        }
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
    return std::make_tuple(success,loc,offset);
  }
}


std::tuple<std::size_t,bool> gpufortrt::internal::record_list_t::find_available_record(std::size_t num_bytes) {
  std::size_t loc = this->records.size(); // element after last record
  bool reuse_existing = false;
  for ( std::size_t i = 0; i < this->records.size(); i++ ) {
    auto& record = this->records[i];
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
  return std::make_tuple(loc,reuse_existing);
}

namespace {
  bool verify_is_proper_section_of_host_data(
     gpufortrt::internal::record_t& record,
     void* hostptr,
     std::size_t num_bytes,
     bool check_restrictions = true) {
    auto tup/*is_subset,offset*/ = gpufortrt::internal::is_host_data_subset(record,hostptr,num_bytes);
    if ( std::get<0>(tup) ) {
      return true;
    } else if ( check_restrictions ) {
      std::stringstream ss;
      ss << "host data to map (" << hostptr << " x " << num_bytes << " B) overlaps with but is no subset of already existing record's host data ("
         << record.hostptr << " x " << record.num_bytes << " B); "
         << "(hostptr - record.hostptr) = " << std::get<1>(tup) << " B";
      LOG_ERROR(ss.str())
    }
    return false;
  }
} // namespace

// implements the data clause restriction
std::tuple<bool,std::size_t> gpufortrt::internal::record_list_t::increment_record_if_present(
   gpufortrt_counter_t ctr_to_update,
   void* hostptr,
   std::size_t num_bytes,
   bool check_restrictions) {
  auto list_tuple/*success,loc,offset*/ = this->find_record(hostptr);
  const bool& found_candidate = std::get<0>(list_tuple);
  const size_t& loc = std::get<1>(list_tuple);
  //
  bool record_present = false;
  if ( found_candidate ) {
    auto& record = this->records[loc];
    if ( ::verify_is_proper_section_of_host_data(record,hostptr,num_bytes,check_restrictions) ) {
      record.inc_refs(ctr_to_update);
      record_present = true;
    }
  }
  return std::make_tuple(record_present,loc);
}

std::size_t gpufortrt::internal::record_list_t::create_increment_record(
   gpufortrt_counter_t ctr_to_update,
   void* hostptr,
   std::size_t num_bytes,
   bool never_deallocate,
   bool allocate_device_buffer,
   bool copy_to_device,
   bool blocking,
   gpufortrt_queue_t queue) {
  auto inc_tuple/*present,loc*/ = this->increment_record_if_present(
          ctr_to_update,hostptr,num_bytes,true/*check ...*/);
  const bool& present = std::get<0>(inc_tuple);
  const std::size_t& loc = std::get<1>(inc_tuple);
  if ( !present ) { 
    auto avail_tuple/*loc,reuse*/ = this->find_available_record(num_bytes);
    const std::size_t& loc = std::get<0>(avail_tuple);
    const bool& reuse_existing = std::get<1>(avail_tuple);
    //
    if ( loc == this->records.size() ) {
      this->records.emplace_back();
    }
    auto& record = this->records[std::get<0>(avail_tuple)];
    record.setup(
      gpufortrt::internal::num_records++, // increment global not thread-safe
      hostptr,
      num_bytes,
      allocate_device_buffer,
      copy_to_device,
      blocking,
      queue,
      reuse_existing); // TODO reuse_existing could negate allocate_device_buffer

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
  }
  return loc;
}

void gpufortrt::internal::record_list_t::decrement_release_record(
  gpufortrt_counter_t ctr_to_update,
  void* hostptr,
  std::size_t num_bytes,
  bool copyout,
  bool finalize,
  bool blocking,
  gpufortrt_queue_t queue) {
  std::ptrdiff_t offset = -1;
  auto list_tuple/*success,loc,offset*/ = this->find_record(hostptr);
  const bool& success = std::get<0>(list_tuple);
  const size_t& loc = std::get<1>(list_tuple);
  //
  if ( success ) {
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
