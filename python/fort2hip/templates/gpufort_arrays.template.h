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
  enum class SyncMode {
    None            = 0, //> No copies between host and device data after initialization and before destruction.
    Copy            = 1, //> Copy to device after initialization and back to host before destruction.
    Copyin          = 2, //> Copy to device after initialization.
    Copyout         = 3, //> Copy from device to host before destruction.
    InvertedCopy    = 4, //> 'Copy'   in opposite direction: Copy from device to host after initialization and back to device before destruction.
    InvertedCopyin  = 5, //> 'Copyin' in opposite direction: Copy from device to host after initialization.
    InvertedCopyout = 6  //> 'Copyout'in opposite direction: Copy from host to device before destruction.
  };

  enum class AllocMode {
    WrapHostWrapDevice         = 0, //> Wrap host and device pointers.
    WrapHostAllocDevice        = 1, //> Wrap host pointer and allocate a new device array.
    AllocHostWrapDevice        = 2, //> Allocate new host array and wrap device pointer.
    AllocHostAllocDevice       = 3, //> Allocate new host and device arrays.
    AllocPinnedHostWrapDevice  = 4, //> Allocate new pinned host array and wrap device pointer.
    AllocPinnedHostAllocDevice = 5  //> Allocate new pinned host array and wrap device pointer.
  };

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
    array_descr{{rank}}<T> data;
    AllocMode              alloc_mode = AllocMode::WrapHostAllocDevice; //> Data allocation strategy. Default: 
                                                                        //> wrap the host and allocate device data
    SyncMode               sync_mode  = SyncMode::None;                 //> How data should be synchronized
                                                                        //> during the initialization and destruction of this GPUFORT array.
    int num_refs           = 0;         //> Number of references.
    int bytes_per_element  = -1;        //> Bytes per element; stored to make num_data_bytes routine independent of T 

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
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None,
        hipStream_t stream   = nullptr) {
      this->data.wrap(
          data_host,
          data_dev,
{{ gm.bound_args_single_line("",rank) | indent(10,True) }}
      );
      this->bytes_per_element = bytes_per_element;
      this->num_refs          = 1;
      this->alloc_mode        = alloc_mode;
      this->sync_mode         = sync_mode;
  
      // host array allocation
      hipError_t ierr = hipSuccess;
      switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
#endif
        case AllocMode::AllocPinnedHostWrapDevice:
        case AllocMode::AllocPinnedHostAllocDevice:
          ierr = hipHostMalloc((void**) &this->data.data_host,this->num_data_bytes(),0);
	  break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
          this->data.data_host = (T*) malloc(this->num_data_bytes());
	  break;
#endif
	case AllocMode::WrapHostWrapDevice:
        case AllocMode::WrapHostAllocDevice:
          // do nothing as already set by data.wrap call
	  break;
	default:
          std::cerr << "ERROR: gpufort::Array{{rank}}::init(...): Unexpected value for 'this->alloc_mode': " 
                    << static_cast<int>(this->alloc_mode) << std::endl; 
          std::terminate();
          break;
      }
      // device array allocation
      if ( ierr == hipSuccess ) {
        switch (this->alloc_mode) {
          case AllocMode::AllocHostWrapDevice:
          case AllocMode::AllocPinnedHostWrapDevice:
          case AllocMode::WrapHostWrapDevice:
            // do nothing as already set by data.wrap call
            break;
          case AllocMode::WrapHostAllocDevice:
          case AllocMode::AllocHostAllocDevice:
          case AllocMode::AllocPinnedHostAllocDevice:
            ierr = hipMalloc((void**) &this->data.data_dev, this->num_data_bytes());
            break;
          default:
            break;
        }
      }
      if ( ierr == hipSuccess ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::None:
          case SyncMode::Copyout:
          case SyncMode::InvertedCopyout:
            // do nothing
            break;
          case SyncMode::Copy:
          case SyncMode::Copyin:
            ierr = this->copy_to_device(stream);
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyin:
            ierr = this->copy_to_host(stream);
            break;
          default:
            std::cerr << "ERROR: gpufort::Array{{rank}}::destroy(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(sync_mode) << std::endl; 
            std::terminate();
            break;
        }
      }
      return ierr;
    }
    
    __host__ hipError_t init(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int* sizes,
        int* lower_bounds,
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None,
        hipStream_t stream   = nullptr) {
      return this->init(
        bytes_per_element,
        data_host,
        data_dev,
        sizes[0],{% for d in range(1,rank) %}sizes[{{d}}],{% endfor %}{{""}}
        lower_bounds[0],{% for d in range(1,rank) %}lower_bounds[{{d}}],{% endfor %}{{""}}
        alloc_mode,sync_mode,stream);
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
      if ( bytes_per_element > 0 ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::Copy:
          case SyncMode::Copyout:
            ierr = this->copy_to_host(stream);
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyout:
            ierr = this->copy_to_device(stream);
            break;
          case SyncMode::None:
          case SyncMode::Copyin:
          case SyncMode::InvertedCopyin:
            // do nothing
            break;
          default:
            std::cerr << "ERROR: gpufort::Array{{rank}}::destroy(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(sync_mode) << std::endl; 
            std::terminate();
            break;
        }
        // host array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
#endif
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipHostFree(this->data.data_host);
              break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocHostAllocDevice:
              free(this->data.data_host); 
              break;
#endif
            case AllocMode::WrapHostWrapDevice:
            case AllocMode::WrapHostAllocDevice:
              break;
            default:
              std::cerr << "ERROR: gpufort::Array{{rank}}::destroy(...): Unexpected value for 'alloc_mode': " 
                        << static_cast<int>(alloc_mode) << std::endl; 
              std::terminate();
              break;
          }
        }
        // device array allocation
        if ( ierr == hipSuccess ) {
          switch (this->alloc_mode) {
            case AllocMode::AllocHostWrapDevice:
            case AllocMode::AllocPinnedHostWrapDevice:
            case AllocMode::WrapHostWrapDevice:
              // do nothing as already set by data.wrap call
              break;
            case AllocMode::WrapHostAllocDevice:
            case AllocMode::AllocHostAllocDevice:
            case AllocMode::AllocPinnedHostAllocDevice:
              ierr = hipFree(this->data.data_dev);
              break;
            default:
              break;
          }
        }
        if ( ierr == hipSuccess ) {
          this->data.data_host    = nullptr;
          this->data.data_dev     = nullptr;
          this->bytes_per_element = -1; 
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
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
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
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
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
