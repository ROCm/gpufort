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

void gpufortrt::internal::queue_record_list_t::reserve(int capacity) {
  this->records.reserve(capacity); 
}

void gpufortrt::internal::queue_record_list_t::destroy() {
  for (size_t i = 0; i < records.size(); i++) {
    if ( this->records[i].is_initialized() ) {
      this->records[i].destroy();
    }
  }
  this->records.clear();
}

size_t gpufortrt::internal::queue_record_list_t::find_record(int id,bool& success) const {
  success = false;
  size_t loc = 0; // typically size_t is unsigned
  for (size_t i = 0; i < this->records.size(); i++) {
    auto& record = this->records[i];
    if ( record.id == id ) {
      loc = i;
      success = true;
      break;
    }
  }
  return loc;
}

gpufortrt_queue_t gpufortrt::internal::queue_record_list_t::use_create_queue(int id) {
  if ( id > 0 ) { 
    bool success = false;
    size_t loc = this->find_record(id,success/*inout*/); 
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