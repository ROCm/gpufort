// This file was generated from a template via gpufort --gpufort-create-headers
#ifndef _GPUFORT_REDUCTIONS_H_
#define _GPUFORT_REDUCTIONS_H_
// requires that gpufort.h is included beforehand
#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#include "hipcub/hipcub.hpp"
#include <limits>
// reductions
namespace {
  struct reduce_op_mult {
    template <typename T> 
    static __host__ __device__ __forceinline__ T ival() { return (T)1; }
    template <typename T> 
    static __host__ __device__ __forceinline__ void init(T &a) { a = ival<T>(); }
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
      return a * b;
    }
  };
  
  struct reduce_op_add {
    template <typename T> 
    static __host__ __device__ __forceinline__ T ival() { return (T)0; }
    template <typename T>
    static __host__ __device__ __forceinline__ void init(T &a) { a = ival<T>(); }
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
      return a + b;
    }
  };
  
  struct reduce_op_max {
    template <typename T> 
    static __host__ __device__ __forceinline__ T ival() {
      return -std::numeric_limits<T>::max(); // has negative sign
    }
    template <typename T> 
    static __host__ __device__ __forceinline__ void init(T &a) { a = ival<T>(); }
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
      return std::max(a, b);
    }
  };
  
  struct reduce_op_min {
    template <typename T>
    static __host__ __device__ __forceinline__ T ival() {
      return std::numeric_limits<T>::max();
    }
    template <typename T>
    static __host__ __device__ __forceinline__ void init(T &a) { a = ival<T>(); }
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
      return std::min(a, b);
    }
  };
  
  template <typename T, typename ReduceOpT>
  void reduce(const T *const d_in, const int &NX, const T *h_out) {
    T *d_out = nullptr;
    HIP_CHECK(hipMalloc((void **)&d_out, sizeof(T)));
    // Determine temporary device storage requirements
    void *temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    ReduceOpT reduceOp;
    hipcub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, d_in, d_out, NX,
                                 ReduceOpT(), ReduceOpT::template ival<T>());
    // Allocate temporary storage
    HIP_CHECK(hipMalloc(&temp_storage, temp_storage_bytes));
    // Run reduction
    hipcub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, d_in, d_out, NX,
                                 ReduceOpT(), ReduceOpT::template ival<T>());
    HIP_CHECK(hipMemcpy((void *)h_out, d_out, sizeof(T), hipMemcpyDeviceToHost));
    // Clean up
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipFree(temp_storage));
  }
}
#endif // _GPUFORT_REDUCTIONS_H_