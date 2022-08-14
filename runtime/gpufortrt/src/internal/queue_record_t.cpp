// SPDX-License-Identifier: MIT                                                
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "gpufortrt_core.h"

#include <iostream>

#include "assert.h"

#include "hip/hip_runtime_api.h"

void gpufortrt::internal::queue_record_t::to_string(std::ostream& os) const {
  os << "id: "             << this->id
     << ", initialized: "  << this->is_initialized();
}

std::ostream& operator<<(std::ostream& os,const gpufortrt::internal::queue_record_t& queue_record) 
{
  queue_record.to_string(os); 
  return os;
}
  
bool gpufortrt::internal::queue_record_t::is_initialized() const {
  return this->id > 0;
}

void gpufortrt::internal::queue_record_t::setup(const int id) {
  assert(id > 0);
  assert(!this->is_initialized());
  this->id = id;
  HIP_CHECK(hipStreamCreate(&this->queue))
}

void gpufortrt::internal::queue_record_t::destroy() {
  assert(this->is_initialized());
  HIP_CHECK(hipStreamDestroy(this->queue))
  this->id = -1;
}
