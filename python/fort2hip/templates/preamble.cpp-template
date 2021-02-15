{% if haveReductions %}  
#include "hipcub/hipcub.hpp"
#include <limits>
{% endif %}

#define HIP_CHECK(condition)         \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
  }

// global thread indices for various dimensions
#define __gidx(idx) (threadIdx.idx + blockIdx.idx * blockDim.idx) 
#define __gidx1 __gidx(x)
#define __gidx2 (__gidx(x) + gridDim.x*blockDim.x*__gidx(y))
#define __gidx3 (__gidx(x) + gridDim.x*blockDim.x*__gidx(y) + gridDim.x*blockDim.x*gridDim.y*blockDim.y*__gidx(z))
#define __total_threads(grid,block) ( (grid).x*(grid).y*(grid).z * (block).x*(block).y*(block).z )

namespace {
  template <typename I, typename E, typename S> __device__ __forceinline__ bool loop_cond(I idx,E end,S stride) {
    return (stride>0) ? ( idx <= end ) : ( -idx <= -end );     
  }

{% for floatType in ["float", "double"] %}  // make {{floatType}}
{% for type in ["short int",  "unsigned short int",  "unsigned int",  "int",  "long int",  "unsigned long int",  "long long int",  "unsigned long long int",  "signed char",  "unsigned char",  "float",  "double",  "long double"] %} __device__ __forceinline__ {{floatType}} make_{{floatType}}(const {{type}}& a) {
    return static_cast<{{floatType}}>(a);
  }
{% endfor %}{% for type in ["hipFloatComplex", "hipDoubleComplex" ] %} __device__ __forceinline__ {{floatType}} make_{{floatType}}(const {{type}}& a) {
    return static_cast<{{floatType}}>(a.x);
  }
{% endfor %}
{% endfor %} // conjugate complex type
  __device__ __forceinline__ hipFloatComplex conj(const hipFloatComplex& c) {
    return hipConjf(c);
  }
  __device__ __forceinline__ hipDoubleComplex conj(const hipDoubleComplex& z) {
    return hipConj(z);
  }

  // TODO Add the following functions:
  // - sign(x,y) = sign(y) * |x| - sign transfer function
  // ...
} 

{% if haveReductions %}  
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
    hipMalloc((void **)&d_out, sizeof(T));
    // Determine temporary device storage requirements
    void *temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    ReduceOpT reduceOp;
    hipcub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, d_in, d_out, NX,
                                 ReduceOpT(), ReduceOpT::template ival<T>());
    // Allocate temporary storage
    hipMalloc(&temp_storage, temp_storage_bytes);
    // Run reduction
    hipcub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, d_in, d_out, NX,
                                 ReduceOpT(), ReduceOpT::template ival<T>());
    hipMemcpy((void *)h_out, d_out, sizeof(T), hipMemcpyDeviceToHost);
    // Clean up
    hipFree(d_out);
    hipFree(temp_storage);
  }
}
{% endif %}

// end of preamble
