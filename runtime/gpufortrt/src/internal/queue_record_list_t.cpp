// SPDX-License-Identifier: MIT                                                
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "gpufortrt_core.h"

#include <iostream>

#include "assert.h"

#include "hip/hip_runtime_api.h"

#include "auxiliary.h"

gpufortrt::internal::queue_record_t& gpufortrt::internal::queue_record_list_t::operator[](int i) {
  return this->records[i];  
}

const gpufortrt::internal::queue_record_t& gpufortrt::internal::queue_record_list_t::operator[](int i) const {
  return this->records[i];  
}

size_t gpufortrt::internal::queue_record_list_t::size() {
  return this->records.size();
}

void gpufortrt::internal::queue_record_list_t::reserve(int capacity) {
  this->records.reserve(capacity); 
}

void gpufortrt::internal::queue_record_list_t::destroy() {
  for (std::size_t i = 0; i < records.size(); i++) {
    if ( this->records[i].is_initialized() ) {
      this->records[i].destroy();
    }
  }
  this->records.clear();
}

std::tuple<bool,std::size_t> gpufortrt::internal::queue_record_list_t::find_record(int id) const {
  bool success = false;
  std::size_t loc = 0; // typically std::size_t is unsigned
  for (std::size_t i = 0; i < this->records.size(); i++) {
    auto& record = this->records[i];
    if ( record.id == id ) {
      loc = i;
      success = true;
      break;
    }
  }
  return std::make_tuple(success,loc);
}

gpufortrt_queue_t gpufortrt::internal::queue_record_list_t::use_create_queue(int id) {
  if ( id > 0 ) { 
    bool success;
    std::size_t loc;
    std::tie(success,loc) = this->find_record(id);
    if ( success ) {
      LOG_INFO(3,"use existing queue; " << this->records[loc])
      return this->records[loc].queue; 
    } else {
      gpufortrt::internal::queue_record_t record;
      record.setup(id);
      LOG_INFO(3,"create queue; " << record)
      this->records.push_back(record);
      return record.queue;
    }
  } else {
    LOG_INFO(3,"use default queue; id:" << id)
    return gpufortrt_default_queue;
  }
}


namespace {
  void synchronize_default_queue() {
    gpufortrt::internal::queue_record_t default_queue_record;
    default_queue_record.setup(-1); // original was (-1,gpufortrt_default_queue)
    default_queue_record.synchronize();
  }
  
  bool test_default_queue() {
    gpufortrt::internal::queue_record_t default_queue_record;
    default_queue_record.setup(-1); // original was (-1,gpufortrt_default_queue)
    return default_queue_record.test();
  }
}

/** \return If all operations in the queue with identifier `id` 
 * have been completed.*/
bool gpufortrt::internal::queue_record_list_t::test(const int id) {
  if ( id > 0 ) {
    bool success;
    std::size_t loc;
    std::tie(success,loc) = this->find_record(id);
    if ( success ) {
      bool result = this->records[loc].test(); 
      LOG_INFO(3,"<test queue; " << this->records[loc] << "; result:"<<result)
      return result;
    } else {
      LOG_INFO(3,"<test queue; no queue found for id="<<id<<"; result:1")
      return true;
    }
  } else {
    bool result = ::test_default_queue();
    LOG_INFO(3,"<test default queue; result:"<< result)
    return result;
  }
}

/** Synchronize queue with identifier `id`. */
void gpufortrt::internal::queue_record_list_t::synchronize(const int id) {
  if ( id > 0 ) {
    bool success;
    std::size_t loc;
    std::tie(success,loc) = this->find_record(id);
    if ( success ) {
      this->records[loc].synchronize(); 
      LOG_INFO(3,"synchronize queue; " << this->records[loc])
    } else {
      LOG_INFO(3,"synchronize queue; no queue found for id="<<id)
    }
  } else {
    LOG_INFO(3,"synchronize default queue; id:" << id)
    ::synchronize_default_queue();
  }
}

/** \return If all operations in all queues
 * have been completed.*/
bool gpufortrt::internal::queue_record_list_t::test_all() {
  bool result = true;
  for (size_t i = 0; i < gpufortrt::internal::queue_record_list.records.size(); i++) {
    auto& queue_record = gpufortrt::internal::queue_record_list[i];
    result &= queue_record.test();
  } 
  result &= ::test_default_queue();
  LOG_INFO(3,"<test all queues; result:" << result)
  return result;
}

/** Synchronize all queues. */
void gpufortrt::internal::queue_record_list_t::synchronize_all() {
  LOG_INFO(3,"synchronize all queues;")
  for (size_t i = 0; i < gpufortrt::internal::queue_record_list.records.size(); i++) {
    auto& queue_record = gpufortrt::internal::queue_record_list[i];
    queue_record.synchronize();
  } 
  ::synchronize_default_queue();
}
