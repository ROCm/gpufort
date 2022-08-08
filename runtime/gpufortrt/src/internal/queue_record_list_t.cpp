// SPDX-License-Identifier: MIT                                                
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "gpufortrt_core.h"

#include <iostream>

#include "assert.h"

#include <hip/hip_runtime_api>

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

size_t gpufortrt::internal::queue_record_list_t::find_record(int id) const {
  size_t loc = -1;
  for (size_t i = 0; i < this->records.size(); i++) {
    auto& record = this->records[i];
    if ( record.id == id ) {
      loc = i;
      break;
    }
  }
  return loc;
}

size_t gpufortrt::internal::queue_record_list_t::find_available_record(int id) const {
  size_t loc = this->records.size();
  //for (size_t i = 0; i < this->records.size(); i++) {
  //  auto& record = this->records[i];
  //  if ( !record.is_initialized() ) {
  //    loc = i;
  //    break;
  //  }
  //}
  return loc;
}

gpufortrt_queue_t gpufortrt_queue_record_list_t::use_create_queue(int id) {
  if ( id > 0 ) { 
    size_t loc = this->find_record(int id); 
    if ( loc >= 0 ) {
      return this->records[loc].queue; 
    } else {
      size_t loc = this->find_available_record(int id);
      if ( loc == this->records.size() ) {
        gpufortrt::internal::queue_record_t record;
        record.setup(id);
        this->records.push_back(record);
      }
      return this->records[loc].queue;
    }
  else {
    return gpufortrt_default_queue;
  }
}
