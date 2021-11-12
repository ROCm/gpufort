{# SPDX-License-Identifier: MIT                                                 #}
{# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved. #}
{% import "templates/gpufort.macros.h" as gm %}
{% import "templates/gpufort_arrays.macros.h" as gam %}
// This file was generated from a template via gpufort --gpufort-create-headers
#ifndef _GPUFORT_ARRAYS_H_
#  define _GPUFORT_ARRAYS_H_
#  include <hip/hip_runtime_api.h>
#  ifndef _GPUFORT_H_
#    include <iostream>
#    include <stdlib.h>
#    define HIP_CHECK(condition)     \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
  }
#  endif
#ifndef __HIP_DEVICE_COMPILE__
#  include <assert.h>
#endif
namespace gpufort {
{% for rank in range(1,max_rank+1) %}
{% set rank_ub = rank+1 %}
  /**
   * Intended to be passed as kernel launch parameter.
   * \note Operator `T& operator()` requires parametrization with type T.
   * \note Size of this struct independent of template type T.
   */
  template<typename T>
  struct array_descr{{rank}} {
    T*     data_host    = nullptr;
    T*     data_dev     = nullptr;
    size_t num_elements = 0;     //> Number of represented by this array.
    int    index_offset = -1;    //> Offset for index calculation; scalar product of negative lower bounds and strides.
{% for d in range(1,rank_ub) %}
    int    stride{{d}}      = -1;    //> Stride {{d}} for linearizing {{rank}}-dimensional index.
{% endfor %}
 
    /**
     * Wrap a host-device pointer pair and store the associated
     * array meta data.
     *
     * \param[in] data_host host data pointer.
     * \param[in] data_dev device data pointer.
     * \param[in] n1,n2,... element counts per dimension of the multi-dim. array 
     * \param[in] lb1,lb2,... lower bounds, i.e. the index of the first element per dimension of the multi-dim. array 
     */ 
    __host__ __device__ void wrap(
        T* data_host,
        T* data_dev,
{{ gm.bound_args("const int ",rank) | indent(8,True) }}
      ) {
       this->data_host = data_host;
       this->data_dev  = data_dev;
       // column-major access
       this->stride1 = 1;
{% for d in range(2,rank_ub) %}
       this->stride{{d}} = this->stride{{d-1}}*n{{d-1}};
{% endfor %}
       this->num_elements = this->stride{{rank}}*n{{rank}};
       this->index_offset =
{% for d in range(1,rank_ub) %}
         -lb{{d}}*this->stride{{d}}{{";" if loop.last}}
{% endfor %}
    }
    
    /**
     * Linearize multi-dimensional index.
     *
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ int linearized_index (
{{ gm.arglist("const int i",rank) | indent(6,"True") }}
    ) {
      return this->index_offset
{% for d in range(1,rank_ub) %}
          + i{{d}}*this->stride{{d}}{{";" if loop.last}}
{% endfor %}
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ T& operator() (
{{ gm.arglist("const int i",rank) | indent(6,"True") }}
    ) {
      const int index = linearized_index(
{{ gm.separated_list_single_line("i",",",rank) | indent(8,"True") }}
      );
      #ifdef __HIP_DEVICE_COMPILE__
      return this->data_dev[index];
      #else
{% for r in range(1,rank_ub) %}
      assert(i{{r}} >= lbound({{r}}));
      assert(i{{r}} <= ubound({{r}}));
{% endfor %}
      return this->data_host[index];
      #endif
    }
    
    /**
     * \return Size of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,{{rank}}
     */
    __host__ __device__ __forceinline__ int size(int dim) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(dim >= 1);
      assert(dim <= {{rank}});
      #endif
      switch(dim) {
       case {{rank}}:
	  return this->num_elements / this->stride{{rank}};
{% for r in range(rank-1,0,-1) %}
       case {{r}}:
	  return this->stride{{r+1}} / this->stride{{r}};
{% endfor %}
       default:
          #ifndef __HIP_DEVICE_COMPILE__
          std::cerr << "‘dim’ argument of ‘gpufort::array{{rank}}::size’ is not a valid dimension index ('dim': "<<dim<<", max dimension: {{rank}}" << std::endl;
          std::terminate();
          #else
          return -1;
          #endif
      }
    }
    
    /**
     * \return Lower bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,{{rank}}
     */
    __host__ __device__ __forceinline__ int lbound(int dim) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(dim >= 1);
      assert(dim <= {{rank}});
      #endif
      int offset_remainder = this->index_offset;
{% for r in range(rank,0,-1) %}
      {{"int " if r == rank}}lb = -offset_remainder / this->stride{{r}};
{% if r == 1 %}
      #ifndef __HIP_DEVICE_COMPILE__
      offset_remainder = offset_remainder + lb*this->stride{{r}};
      assert(offset_remainder == 0);
      #endif
      return lb;
{% else %}
      if ( dim == {{r}} ) return lb;
      offset_remainder = offset_remainder + lb*this->stride{{r}};
{% endif %}
{% endfor %}
    }
    
    /**
     * \return Upper bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,{{rank}}
     */
    __host__ __device__ __forceinline__ int ubound(int dim) {
      return this->lbound(dim) + this->size(dim) - 1;
    }
  };

  template<typename T>
  struct array{{rank}}{
    array_descr{{rank}}<T>      data;
    bool pinned                 = false;     //> If the host data is pinned. 
    bool copyout_at_destruction = false;     //> If the device data should be copied back to the host when this struct is destroyed.
    bool owns_host_data         = false;     //> If this is only a wrapper, i.e. no memory management is performed.
    bool owns_device_data       = false;     //> If this is only a wrapper, i.e. no memory management is performed.
    int num_refs                = 0;         //> Number of references.
    int bytes_per_element       = -1;        //> Bytes per element; stored to make num_data_bytes routine independent of T 

    array{{rank}}() {
      // do nothing
    }
    
    ~array{{rank}}() {
      // do nothing
    }

    /**
     * Initialize.
     * \param[in] data_host host data pointer (may be nullptr; see the note).
     * \param[in] pinned the host array is pinned or should be pinned.
     * \param[in] copyout_at_destruction copy data out from the
     *            device to the host when this instance gets destroyed.
     * \note Allocates a new host_data pointer if the
     * passed pointer is a nullptr. If the pinned argument is true,
     * the new hostpotr is allocated via hipHostMalloc.
     * Otherwise, it is allocated via classic malloc.
     */
    __host__ hipError_t init(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
{{ gm.bound_args_single_line("int ",rank) | indent(8,True) }},
        bool pinned,
        bool copyout_at_destruction = false) {
      hipError_t ierr = hipSuccess;
      this->data.wrap(
          data_host,
          data_dev,
{{ gm.bound_args_single_line("",rank) | indent(10,True) }}
      );
      this->bytes_per_element      = bytes_per_element;
      this->pinned                 = pinned;
      this->copyout_at_destruction = copyout_at_destruction; 
      this->owns_host_data         = data_host == nullptr;
      this->owns_device_data       = data_dev == nullptr;
      this->num_refs               = 1;
      if ( this->owns_host_data && pinned ) {
        ierr = hipHostMalloc((void**) &this->data.data_host,this->num_data_bytes(),0);
      } else if ( this->owns_host_data ) {
        this->data.data_host = (T*) malloc(this->num_data_bytes());
      }
      if ( ierr == hipSuccess && this->owns_device_data ) {
        ierr = hipMalloc((void**) &this->data.data_dev, this->num_data_bytes());
      } 
      return ierr;
    }
    
    __host__ hipError_t init(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int* sizes,
        int* lower_bounds,
        bool pinned,
        bool copyout_at_destruction = false) {
        return this->init(
          bytes_per_element,
          data_host,
          data_dev,
          sizes[0],{% for d in range(1,rank) %}sizes[{{d}}],{% endfor %}{{""}}
          lower_bounds[0],{% for d in range(1,rank) %}lower_bounds[{{d}}],{% endfor %}{{""}}
          pinned,
          copyout_at_destruction);
    }

    /**
     * Destroy associated host and device data
     * if this is not a wrapper. Depending
     * on the pinned attribue, hipHostFree 
     * is used instead of free to deallocate
     * the host data.
     */
    __host__ hipError_t destroy(hipStream_t stream) {
      hipError_t ierr = hipSuccess;
      if ( this->bytes_per_element > 0 ) {
        if ( this->owns_host_data && this->data.data_host != nullptr ) {
          if ( this->pinned ) {
            ierr = hipHostFree(this->data.data_host);
          } else {
            free(this->data.data_host); 
          }
        }
        if ( ierr == hipSuccess ) {
          if ( this->owns_device_data && this->data.data_dev != nullptr ) {
            if ( this->copyout_at_destruction ) {
              this->copy_to_host(stream);
            }
            ierr = hipFree(this->data.data_dev);
          }
        }
      }
      return ierr;
    }
    
    __host__ size_t num_data_bytes() {
      return this->data.num_elements * this->bytes_per_element;
    }
 
    /**
     * Copy host data to the device.
     * \return Array code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_host(hipStream_t stream = nullptr) {
      return hipMemcpyAsync(
        (void*) this->data.data_host, 
        (void*) this->data.data_dev,
        this->num_data_bytes(), 
        hipMemcpyDeviceToHost, stream);
    }
    
    /**
     * Copy device data to the host.
     * \return Array code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_device(hipStream_t stream = nullptr) {
      return hipMemcpyAsync(
        (void*) this->data.data_dev, 
        (void*) this->data.data_host,
        this->num_data_bytes(),
        hipMemcpyHostToDevice, stream);
    }
    
    /** 
     * Copies the struct to the device into an pre-allocated
     * device memory block but does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[in] device_struct device memory address to copy to
     */
    __host__ hipError_t copy_self_to_device(
        gpufort::array{{rank}}<T>* device_struct,
        hipStream_t stream = nullptr
    ) {
      const size_t size = sizeof(array{{rank}}<char>); // sizeof(T*) = sizeof(char*)
      return hipMemcpyAsync(
          (void*) device_struct, 
          (void*) this,
          size,
          hipMemcpyHostToDevice, stream);
    }

    /** 
     * Copies the struct to the device but
     * does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[inout] device_copy pointer to device copy pointer 
     */
    __host__ hipError_t create_device_copy(
        gpufort::array{{rank}}<T>** device_copy,
        hipStream_t stream = nullptr
    ) {
      const size_t size = sizeof(array{{rank}}<char>); // sizeof(T*) = sizeof(char*)
      hipError_t ierr   = hipMalloc((void**)device_copy,size);      
      if ( ierr == hipSuccess ) {
        return this->copy_self_to_device(*device_copy,stream);
      } else {
        return ierr;
      }
    }

    /**
     * Linearize multi-dimensional index.
     *
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ int linearized_index (
{{ gm.arglist("const int i",rank) | indent(6,"True") }}
    ) {
      return this->data.linearized_index(
{{ gm.separated_list_single_line("i",",",rank) | indent(8,"True") }}
      );
    }
    
    /**
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ T& operator() (
{{ gm.arglist("const int i",rank) | indent(6,"True") }}
    ) {
      return this->data(
{{ gm.separated_list_single_line("i",",",rank) | indent(8,"True") }}
      );
    }
    
    /**
     * \return Size of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,{{rank}}
     */
    __host__ __device__ __forceinline__ int size(int dim) {
      return this->data.size(dim);
    }
    
    /**
     * \return Lower bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,{{rank}}
     */
    __host__ __device__ __forceinline__ int lbound(int dim) {
      return this->data.lbound(dim);
    }
    
    /**
     * \return Upper bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,{{rank}}
     */
    __host__ __device__ __forceinline__ int ubound(int dim) {
      return this->data.ubound(dim);
    }
  };
{{ "" if not loop.last }}
{% endfor -%}
}
#endif // _GPUFORT_ARRAYS_H_  
