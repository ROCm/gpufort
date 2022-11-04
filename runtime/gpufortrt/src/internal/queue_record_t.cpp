// SPDX-License-Identifier: MIT                                                
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "gpufortrt_core.h"

#include <iostream>

#include "assert.h"

#include "hip/hip_runtime_api.h"

#include "auxiliary.h"

void gpufortrt::internal::queue_record_t::to_string(std::ostream& os) const {
  os << "id: " << this->id
     << ", queue: "  << this->queue
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
  HIP_CHECK(hipStreamCreate(&this->queue)) // TODO backend-specific, externalize
  LOG_INFO(4,"create queue; " << *this)
}

void gpufortrt::internal::queue_record_t::destroy() {
  LOG_INFO(4,"destroy queue; " << *this)
  assert(this->is_initialized());
  HIP_CHECK(hipStreamDestroy(this->queue)) // TODO backend-specific, externalize
  this->id = -1;
}

bool gpufortrt::internal::queue_record_t::test() {
  bool result = hipStreamQuery(this->queue) == hipSuccess;
  LOG_INFO(4,"<test queue; " << *this <<" ; result:" << result )
  return result;
}

void gpufortrt::internal::queue_record_t::synchronize() {
  LOG_INFO(4,"synchronize queue; " << *this)
  HIP_CHECK(hipStreamSynchronize(this->queue))
}
