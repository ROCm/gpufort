{# SPDX-License-Identifier: MIT                                                 #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{% import "gpufort.macros.h" as gm %}
{% import "gpufort_array.macros.h" as gam %}
// This file was generated from a template via gpufort --gpufort-create-headers
#ifndef _GPUFORT_ARRAYS_H_
#  define _GPUFORT_ARRAYS_H_
#  include <hip/hip_runtime_api.h>
#  ifndef _GPUFORT_H_
#    include <iostream>
#    include <iomanip>
#    include <stdlib.h>
#    include <vector>
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
#ifndef GPUFORT_ARRAY_PRINT_PREC
#  define GPUFORT_ARRAY_PRINT_PREC 6
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
  
  enum class PrintMode {
    PrintNorms                 = 0, //> Print norms and metrics.
    PrintValues                = 1, //> Print array values; write index in Fortran (i,j,k,...) style.
    PrintValuesAndNorms        = 2, //> Print array values and norms and metrics; write index in Fortran (i,j,k,...) style.
    PrintCValues               = 3, //> Print array values; write index in C/C++ [i] style.
    PrintCValuesAndNorms       = 4  //> Print array values and norms and metrics; write index in C/C++ [i] style.
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
    int num_elements    = 0;     //> Number of elements represented by this array.
    int    index_offset = -1;    //> Offset for index calculation; scalar product of negative lower bounds and strides.
{% for d in range(1,rank_ub) %}
    int    stride{{d}}  = -1;    //> Stride {{d}} for linearizing {{rank}}-dimensional index.
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
    ) const {
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
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ const T& operator() (
{{ gm.arglist("const int i",rank) | indent(6,"True") }}
    ) const {
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
    __host__ __device__ __forceinline__ int size(int dim) const {
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
    __host__ __device__ __forceinline__ int lbound(int dim) const {
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
    __host__ __device__ __forceinline__ int ubound(int dim) const {
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
    int num_refs             = 0;  //> Number of references.
    size_t bytes_per_element = -1; //> Bytes per element; stored to make num_data_bytes routine independent of T 

    array{{rank}}() {
      // do nothing
    }
    
    ~array{{rank}}() {
      // do nothing
    }

{% for async_suffix in ["_async",""] %}
{% set is_async = async_suffix == "_async" %}
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
    __host__ hipError_t init{{async_suffix}}(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
{{ gm.bound_args_single_line("int ",rank) | indent(8,True) }},{{"
        hipStream_t stream," if is_async}}
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
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
          std::cerr << "ERROR: gpufort::array{{rank}}::init{{async_suffix}}(...): Unexpected value for 'this->alloc_mode': " 
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
            ierr = this->copy_to_device{{async_suffix}}({{"stream" if is_async}});
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyin:
            ierr = this->copy_to_host{{async_suffix}}({{"stream" if is_async}});
            break;
          default:
            std::cerr << "ERROR: gpufort::array{{rank}}::init{{async_suffix}}(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
            std::terminate();
            break;
        }
      }
      return ierr;
    }
   
    __host__ hipError_t init{{async_suffix}}(
        int bytes_per_element,
        T* data_host,
        T* data_dev,
        int* sizes,
        int* lower_bounds,{{"
        hipStream_t stream," if is_async}}
        AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
        SyncMode sync_mode   = SyncMode::None
    ) {
      return this->init{{async_suffix}}(
        bytes_per_element,
        data_host,
        data_dev,
        sizes[0],{% for d in range(1,rank) %}sizes[{{d}}],{% endfor %}{{""}}
        lower_bounds[0],{% for d in range(1,rank) %}lower_bounds[{{d}}],{% endfor %}{{""}}{{"
        stream," if is_async}}alloc_mode,sync_mode);
    }
    
    /**
     * Destroy associated host and device data
     * if this is not a wrapper. Depending
     * on the pinned attribue, hipHostFree 
     * is used instead of free to deallocate
     * the host data.
     */
    __host__ hipError_t destroy{{async_suffix}}({{"hipStream_t stream" if is_async}}) {
      hipError_t ierr = hipSuccess;
      if ( bytes_per_element > 0 ) {
        // synchronize host/device
        switch (this->sync_mode) {
          case SyncMode::Copy:
          case SyncMode::Copyout:
            ierr = this->copy_to_host{{async_suffix}}({{"stream"if is_async}});
            break;
          case SyncMode::InvertedCopy:
          case SyncMode::InvertedCopyout:
            ierr = this->copy_to_device{{async_suffix}}({{"stream"if is_async}});
            break;
          case SyncMode::None:
          case SyncMode::Copyin:
          case SyncMode::InvertedCopyin:
            // do nothing
            break;
          default:
            std::cerr << "ERROR: gpufort::array{{rank}}::destroy{{async_suffix}}(...): Unexpected value for 'sync_mode': " 
                      << static_cast<int>(this->sync_mode) << std::endl; 
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
              std::cerr << "ERROR: gpufort::array{{rank}}::destroy{{async_suffix}}(...): Unexpected value for 'alloc_mode': " 
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
    
    /**
     * Copy data to a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the destinaton buffer is interpreted as host or device variable
     * and the source is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_to_buffer{{async_suffix}}(
       void* buffer, hipMemcpyKind memcpy_kind{{",
       hipStream_t stream" if is_async}}
    ) {
      T* src = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyDeviceToHost: 
        case hipMemcpyDeviceToDevice:
          src = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyHostToDevice:
          src = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array{{rank}}::copy_to_buffer{{async_suffix}}(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(src!=nullptr);
      #endif
      return hipMemcpy{{"Async" if is_async}}(
        buffer, 
        (void*) src,
        this->num_data_bytes(), 
        memcpy_kind{{", stream" if is_async}});
    }
    
    /**
     * Copy data from a buffer.
     * Depending on the hipMemcpyKind parameter,
     * the source buffer is interpreted as host or device variable
     * and the destination is the array's host or device buffer.
     * \return  Error code returned by the underlying hipMemcpy operation.
     * \note Use num_data_bytes() for determining the buffer size.
     */
    __host__ hipError_t copy_from_buffer{{async_suffix}}(
       void* buffer, hipMemcpyKind memcpy_kind{{",
       hipStream_t stream" if is_async}}
    ) {
      T* dest = nullptr ;
      switch (memcpy_kind) {
        case hipMemcpyHostToDevice: 
        case hipMemcpyDeviceToDevice:
          dest = this->data.data_dev;
          break;
        case hipMemcpyHostToHost: 
        case hipMemcpyDeviceToHost:
          dest = this->data.data_host;
          break;
        default:
          std::cerr << "ERROR: gpufort::array{{rank}}::copy_from_buffer{{async_suffix}}(...): Unexpected value for 'memcpy_kind': " 
                    << static_cast<int>(memcpy_kind) << std::endl; 
          std::terminate();
          break;
      }
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      assert(dest!=nullptr);
      #endif
      return hipMemcpy{{"Async" if is_async}}(
        (void*) dest,
        buffer, 
        this->num_data_bytes(), 
        memcpy_kind{{", stream" if is_async}});
    }
    
    /**
     * Copy host data to the device.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_host{{async_suffix}}({{"hipStream_t stream" if is_async}}) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpy{{"Async" if is_async}}(
        (void*) this->data.data_host, 
        (void*) this->data.data_dev,
        this->num_data_bytes(), 
        hipMemcpyDeviceToHost{{", stream" if is_async}});
    }
    
    /**
     * Copy device data to the host.
     * \return  Error code returned by the underlying hipMemcpy operation.
     */
    __host__ hipError_t copy_to_device{{async_suffix}}({{"hipStream_t stream" if is_async}}) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(this->data.data_host!=nullptr);
      assert(this->data.data_dev!=nullptr);
      #endif
      return hipMemcpy{{"Async" if is_async}}(
        (void*) this->data.data_dev, 
        (void*) this->data.data_host,
        this->num_data_bytes(),
        hipMemcpyHostToDevice{{", stream" if is_async}});
    }
    
    /** 
     * Copies the struct to the device into an pre-allocated
     * device memory block but does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[in] device_struct device memory address to copy to
     */
    __host__ hipError_t copy_self_to_device{{async_suffix}}(
        gpufort::array{{rank}}<T>* device_struct{{",
        hipStream_t stream" if is_async}}
    ) {
      const size_t size = sizeof(array{{rank}}<char>); // sizeof(T*) = sizeof(char*)
      return hipMemcpy{{"Async" if is_async}}(
          (void*) device_struct, 
          (void*) this,
          size,
          hipMemcpyHostToDevice{{", stream" if is_async}});
    }

    /** 
     * Copies the struct to the device but
     * does not touch the data associated with 
     * data_dev & data_host (shallow copy).
     * \param[inout] device_copy pointer to device copy pointer 
     */
    __host__ hipError_t create_device_copy{{async_suffix}}(
        gpufort::array{{rank}}<T>** device_copy{{",
        hipStream_t stream" if is_async}}
    ) {
      const size_t size = sizeof(array{{rank}}<char>); // sizeof(T*) = sizeof(char*)
      hipError_t ierr   = hipMalloc((void**)device_copy,size);      
      if ( ierr == hipSuccess ) {
        return this->copy_self_to_device{{async_suffix}}(*device_copy{{",stream" if is_async}});
      } else {
        return ierr;
      }
    }
{% endfor %}

{% for target in ["host","device"] %}
{% set is_host = target == "host" %}
    /**
     * Allocate a {{target}} buffer with
     * the same size as the data buffers
     * associated with this gpufort array.
     * @see num_data_bytes()
     * \param[inout] pointer to the buffer to allocate
{% if is_host %}
     * \param[in] pinned If the memory should be pinned (default=True)
     * \param[in] flags  Flags for the host memory allocation (default=0).
{% endif %}
     */
    __host__ hipError_t allocate_{{target}}_buffer(void** buffer{{",bool pinned=true,int flags=0" if is_host}}) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
{% if target == "device" %} 
      ierr = hipMalloc(buffer,this->num_data_bytes());
{% else %}
      if ( pinned ) { 
        ierr = hipHostMalloc(buffer,this->num_data_bytes(),flags);
      } else {
        *buffer = malloc(this->num_data_bytes());
      }
{% endif %}
      return ierr;
    }

    /**
     * Deallocate a {{target}} buffer
     * created via the allocate_{{target}}_buffer routine.
     * @see num_data_bytes(), allocate_{{target}}_buffer
     * \param[inout] the buffer to deallocte
{% if is_host %}
     * \param[in] pinned If the memory to deallocate is pinned (default=True)
{% endif %}
     */
    __host__ hipError_t deallocate_{{target}}_buffer(void* buffer{{",bool pinned=true" if is_host}}) {
      #ifndef __HIP_DEVICE_COMPILE__
      assert(buffer!=nullptr);
      #endif
      hipError_t ierr = hipSuccess;
{% if target == "device" %} 
      ierr = hipFree(buffer);
{% else %}
      if ( pinned ) { 
        ierr = hipHostFree(buffer);
      } else {
        free(buffer);
      }
{% endif %}
      return ierr;
    }
{% endfor %}

    /** 
     * Copy metadata from another gpufort array to this gpufort array.
     * \note The allocation mode is always set to AllocMode::WrapHostWrapDevice
     *       and the sync mode is set to SyncMode::None.
     * \note No deep copy, i.e. the host and device data is not copied, just the pointers
     * to the data.
     */
    __host__ void copy(const gpufort::array{{rank}}<T>& other) {
      this->init(
        other.bytes_per_element,
        other.data.data_host,
        other.data.data_dev,
        other.size(1),{% for d in range(2,rank_ub) %}other.size({{d}}),{% endfor %}{{""}}
        other.lbound(1),{% for d in range(2,rank_ub) %}other.lbound({{d}}),{% endfor %}{{""}}
        AllocMode::WrapHostWrapDevice,
        SyncMode::None);
    }

    /** 
     *  \return Number of bytes required to store 
     *  the array elements on the host or the device.
     */ 
    __host__ size_t num_data_bytes() {
      return this->data.num_elements * this->bytes_per_element;
    }
    
    /**
     * \return the number of elements of this array.
     */
    __host__ int num_elements() {
      return this->data.num_elements;
    }

    /**
     * Linearize multi-dimensional index.
     *
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ int linearized_index (
{{ gm.arglist("const int i",rank) | indent(6,"True") }}
    ) const {
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
     * \return Element at given index.
     * \param[in] i1,i2,... multi-dimensional array index.
     */
    __host__ __device__ __forceinline__ const T& operator() (
{{ gm.arglist("const int i",rank) | indent(6,"True") }}
    ) const {
      return this->data(
{{ gm.separated_list_single_line("i",",",rank) | indent(8,"True") }}
      );
    }
    
    /**
     * \return Size of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,{{rank}}
     */
    __host__ __device__ __forceinline__ int size(int dim) const {
      return this->data.size(dim);
    }
    
    /**
     * \return Lower bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,{{rank}}
     */
    __host__ __device__ __forceinline__ int lbound(int dim) const {
      return this->data.lbound(dim);
    }
    
    /**
     * \return Upper bound (inclusive) of the array in dimension 'dim'.
     * \param[in] dim selected dimension: 1,...,{{rank}}
     */
    __host__ __device__ __forceinline__ int ubound(int dim) const {
      return this->data.ubound(dim);
    }
    

{% for d in range(rank-1,0,-1) %}
    /**
     * Collapse the array by fixing {{rank-d}} indices.
     * \return A gpufort array of rank {{d}}.
     * \param[in] i{{d+1}},...,i{{rank}} indices to fix.
     */
    __host__ gpufort::array{{d}}<T> collapse(
{% for e in range(d+1,rank_ub) %}
        const int i{{e}}{{"," if not loop.last}}
{% endfor %}
    ) const {
{% for e in range(1,d+1) %}
      const int n{{e}}  = this->size({{e}});
      const int lb{{e}} = this->lbound({{e}});
{% endfor %}
      const int index = linearized_index(
{{"" | indent(8,True)}}{% for e in range(1,d+1) %}lb{{e}},{{"\n" if loop.last}}{%- endfor %}
{{"" | indent(8,True)}}{% for e in range(d+1,rank_ub) %}i{{e}}{{"," if not loop.last}}{%- endfor %});
      T* data_host_new = nullptr; 
      if (this->data.data_host != nullptr) {
        data_host_new = this->data.data_host + index;
      }
      T* data_dev_new = nullptr; 
      if (this->data.data_dev != nullptr) {
        data_dev_new = this->data.data_dev + index;
      }
      gpufort::array{{d}}<T> collapsed_array;
      hipError_t ierr = 
        collapsed_array.init(
          this->bytes_per_element,
          data_host_new,
          data_dev_new,
{{"" | indent(10,True)}}{% for e in range(1,d+1) %}n{{e}},{{"\n" if loop.last}}{%- endfor %}
{{"" | indent(10,True)}}{% for e in range(1,d+1) %}lb{{e}},{{"\n" if loop.last}}{%- endfor %}
          AllocMode::WrapHostWrapDevice,
          SyncMode::None);
      return collapsed_array;
    }
{% endfor %}

{% for source in ["host","device"] %}
    /**
     * Write {{source}} data values and their corresponding index
     * to an output stream.
     * \param[inout] out        output stream
     * \param[in]    prefix     prefix to put before each output line
     * \param[in]    print_prec precision to use when printing floating point numbers 
     *               [default=GPUFORT_ARRAY_PRINT_PREC[default=6]]
     * \note This method only makes sense for numeric data types and
     *       may result in compilation, runtime errors or undefined output if the
     *       underlying data is not numeric.
     */
    __host__ void print_{{source}}_data(
        std::ostream& out, const char* prefix, PrintMode print_mode,
        const int print_prec=GPUFORT_ARRAY_PRINT_PREC) const {
      const bool print_norms = 
	print_mode==PrintMode::PrintNorms ||
        print_mode==PrintMode::PrintValuesAndNorms ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      const bool print_values = 
        print_mode==PrintMode::PrintValues ||
        print_mode==PrintMode::PrintValuesAndNorms;
      const bool print_values_c_style = 
        print_mode==PrintMode::PrintCValues ||
        print_mode==PrintMode::PrintCValuesAndNorms;
      if ( !print_norms && !print_values && !print_values_c_style ) {
        std::cerr << "ERROR: gpufort::array{{rank}}::print_{{source}}_data(...): Unexpected value for 'print_mode': " 
                  << static_cast<int>(print_mode) << std::endl; 
        std::terminate();
      }
{% for col in range(1,rank_ub) %}
      const int n{{col}}  = this->size({{col}});
      const int lb{{col}} = this->lbound({{col}}); 
{% endfor %}
      const int n = this->data.num_elements; 
{% if source == "device" %}
      std::vector<T> host_array(n);
      T* A_h = host_array.data();
      T* A   = this->data.data_dev;
      HIP_CHECK(hipMemcpy(A_h, A, n*sizeof(T), hipMemcpyDeviceToHost));
{% else %}
      T* A_h   = this->data.data_host;
{% endif %}
      T min = +std::numeric_limits<T>::max();
      T max = -std::numeric_limits<T>::max();
      T sum = 0;
      T l1  = 0;
      T l2  = 0;
      out << prefix << ":\n";
{% for col in range(1,rank_ub) %}
    for ( int i{{rank_ub-col}} = 0; i{{rank_ub-col}} < n{{rank_ub-col}}; i{{rank_ub-col}}++ ) {
{% endfor %}
      const int idx = this->linearized_index({% for col in range(1,rank_ub) -%}i{{col}}{{ "," if not loop.last }}{% endfor -%});
      T value = A_h[idx];
      if ( print_norms ) {
        min  = std::min(value,min);
        max  = std::max(value,max);
        sum += value;
        l1  += std::abs(value);
        l2  += value*value;
      }
      if ( print_values ) {
          out << prefix << "(" << {% for col in range(1,rank_ub) -%}(lb{{col}}+i{{col}}) << {{ "\",\" <<" |safe if not loop.last }} {% endfor -%} ") = " << std::setprecision(print_prec) << value << "\n";
      } else if ( print_values_c_style ) {
          out << prefix << "[" << idx << "] = " << std::setprecision(print_prec) << value << "\n";
      }
    {%+ for col in range(1,rank_ub) -%}}{%- endfor %} // for loops
    if ( print_norms ) {
      out << prefix << ":min=" << std::setprecision(print_prec) << min << "\n";
      out << prefix << ":max=" << std::setprecision(print_prec) << max << "\n";
      out << prefix << ":sum=" << std::setprecision(print_prec) << sum << "\n";
      out << prefix << ":l1="  << std::setprecision(print_prec) << l1  << "\n";
      out << prefix << ":l2="  << std::setprecision(print_prec) << std::sqrt(l2) << "\n";
      out << prefix << "num_elements=" << std::setprecision(print_prec) << n << "\n";
    }
  }
{% endfor %}

  };
{{ "" if not loop.last }}
{% endfor -%}
}
#endif // _GPUFORT_ARRAYS_H_