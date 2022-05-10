{# SPDX-License-Identifier: MIT                                                 #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{########################################################################################}
{% import "gpufort.macros.h" as gm %}
{% import "gpufort_array_ptr.macros.h" as gapm %}
{% import "gpufort_array.macros.h" as gam %}
{########################################################################################}
{%- macro render_header_begin() -%}
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
    gpufort::error error = condition;    \
    if(error != gpufort::success){         \
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
{%- endmacro -%}
{########################################################################################}
{%- macro render_header_end() -%}
#endif // _GPUFORT_ARRAYS_H_
{%- endmacro -%}
{########################################################################################}
{%- macro render_print_routines(rank) -%}
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
GPUFORT_HOST_ROUTINE void print_{{source}}_data(
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
  T* A   = this->data_dev.data;
  HIP_CHECK(hipMemcpy(A_h, A, n*sizeof(T), hipMemcpyDeviceToHost));
{% else %}
  T* A_h   = this->data_host;
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
{%- endmacro -%}
{########################################################################################}
{%- macro render_init_routines(rank,async_suffix) -%}
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
GPUFORT_HOST_ROUTINE gpufort::error init{{async_suffix}}(
    int bytes_per_element,
    T* data_host,
    T* data_dev,
{{ gm.separated_list("const int n",",",rank) | indent(2,True) }},
{{ gm.separated_list("const int lb",",",rank) | indent(2,True) }},{{"
    gpufort::queue stream," if is_async}}
    AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
    SyncMode sync_mode   = SyncMode::None
) {
  this->data_host = data_host;
  this->data_dev.init(
      bytes_per_element,
      data_dev,
{{ gm.separated_list_single_line("n",",",rank) | indent(2,True) }},
{{ gm.separated_list_single_line("lb",",",rank) | indent(2,True) }}
  );
  this->alloc_mode        = alloc_mode;
  this->sync_mode         = sync_mode;
  this->num_refs          = 1;
  
  // host array allocation
  gpufort::error ierr = gpufort::success;
  switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
    case AllocMode::AllocHostWrapDevice:
    case AllocMode::AllocHostAllocDevice:
#endif
    case AllocMode::AllocPinnedHostWrapDevice:
    case AllocMode::AllocPinnedHostAllocDevice:
      ierr = hipHostMalloc((void**) &this->data_host,this->size_in_bytes(),0);
	  break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
    case AllocMode::AllocHostWrapDevice:
    case AllocMode::AllocHostAllocDevice:
      this->data_host = (T*) malloc(this->size_in_bytes());
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
  if ( ierr == gpufort::success ) {
    switch (this->alloc_mode) {
      case AllocMode::AllocHostWrapDevice:
      case AllocMode::AllocPinnedHostWrapDevice:
      case AllocMode::WrapHostWrapDevice:
        // do nothing as already set by data.wrap call
        break;
      case AllocMode::WrapHostAllocDevice:
      case AllocMode::AllocHostAllocDevice:
      case AllocMode::AllocPinnedHostAllocDevice:
        ierr = hipMalloc((void**) &this->data_dev.data, this->size_in_bytes());
        break;
      default:
        break;
    }
  }
  if ( ierr == gpufort::success ) {
    // synchronize host/device
    switch (this->sync_mode) {
      case SyncMode::None:
      case SyncMode::Copyout:
      case SyncMode::InvertedCopyout:
        // do nothing
        break;
      case SyncMode::Copy:
      case SyncMode::Copyin:
        ierr = this->copy_data_to_device{{async_suffix}}({{"stream" if is_async}});
        break;
      case SyncMode::InvertedCopy:
      case SyncMode::InvertedCopyin:
        ierr = this->copy_data_to_host{{async_suffix}}({{"stream" if is_async}});
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
   
GPUFORT_HOST_ROUTINE gpufort::error init{{async_suffix}}(
    int bytes_per_element,
    T* data_host,
    T* data_dev,
    int* sizes,
    int* lbounds,{{"
    gpufort::queue stream," if is_async}}
    AllocMode alloc_mode = AllocMode::WrapHostAllocDevice,
    SyncMode sync_mode   = SyncMode::None
) {
  return this->init{{async_suffix}}(
    bytes_per_element,
    data_host,
    data_dev,
    sizes[0],{% for d in range(1,rank) %}sizes[{{d}}],{% endfor %}{{""}}
    lbounds[0],{% for d in range(1,rank) %}lbounds[{{d}}],{% endfor %}{{""}}{{"
    stream," if is_async}}alloc_mode,sync_mode);
}

/**
 * Destroy associated host and device data
 * if this is not a wrapper. Depending
 * on the pinned attribue, hipHostFree 
 * is used instead of free to deallocate
 * the host data.
 */
GPUFORT_HOST_ROUTINE gpufort::error destroy{{async_suffix}}({{"gpufort::queue stream" if is_async}}) {
  gpufort::error ierr = gpufort::success;
  if ( bytes_per_element > 0 ) {
    // synchronize host/device
    switch (this->sync_mode) {
      case SyncMode::Copy:
      case SyncMode::Copyout:
        ierr = this->copy_data_to_host{{async_suffix}}({{"stream"if is_async}});
        break;
      case SyncMode::InvertedCopy:
      case SyncMode::InvertedCopyout:
        ierr = this->copy_data_to_device{{async_suffix}}({{"stream"if is_async}});
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
    if ( ierr == gpufort::success ) {
      switch (this->alloc_mode) {
#ifdef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
#endif
        case AllocMode::AllocPinnedHostWrapDevice:
        case AllocMode::AllocPinnedHostAllocDevice:
          ierr = hipHostFree(this->data_host);
          break;
#ifndef GPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocHostAllocDevice:
          free(this->data_host); 
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
    if ( ierr == gpufort::success ) {
      switch (this->alloc_mode) {
        case AllocMode::AllocHostWrapDevice:
        case AllocMode::AllocPinnedHostWrapDevice:
        case AllocMode::WrapHostWrapDevice:
          // do nothing as already set by data.wrap call
          break;
        case AllocMode::WrapHostAllocDevice:
        case AllocMode::AllocHostAllocDevice:
        case AllocMode::AllocPinnedHostAllocDevice:
          ierr = hipFree(this->data_dev.data);
          break;
        default:
          break;
      }
    }
    if ( ierr == gpufort::success ) {
      this->data_host    = nullptr;
      this->data_dev.data     = nullptr;
      this->bytes_per_element = -1; 
    } 
  }
  return ierr;
}
{%- endmacro -%}
{########################################################################################}
{%- macro render_memory_operations(rank,async_suffix) -%}
{% set is_async = async_suffix == "_async" %}
/**
 * Copy data to a buffer.
 * Depending on the gpufort::Direction parameter,
 * the destinaton buffer is interpreted as host or device variable
 * and the source is the array's host or device buffer.
 * \return  Error code returned by the underlying hipMemcpy operation.
 * \note Use size_in_bytes() for determining the buffer size.
 */
GPUFORT_HOST_ROUTINE gpufort::error copy_data_to_buffer{{async_suffix}}(
   void* buffer, gpufort::Direction direction{{",
   gpufort::queue stream" if is_async}}
) {
  T* src = nullptr ;
  hipMemcpyKind memcpy_kind = hipMemcpyDefault;
  switch (direction) {
    case DeviceToHost: 
      src = this->data_dev.data;
      memcpy_kind = hipMemcpyDeviceToHost;
      break;
    case DeviceToDevice:
      src = this->data_dev.data;
      memcpy_kind = hipMemcpyDeviceToDevice;
      break;
    case HostToHost: 
      src = this->data_host;
      memcpy_kind = hipMemcpyHostToHost;
      break;
    case HostToDevice:
      src = this->data_host;
      memcpy_kind = hipMemcpyHostToDevice;
      break;
    default:
      std::cerr << "ERROR: gpufort::array{{rank}}::copy_data_to_buffer{{async_suffix}}(...): Unexpected value for 'direction': " 
                << static_cast<int>(direction) << std::endl; 
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
    this->size_in_bytes(), 
    memcpy_kind{{", stream" if is_async}});
}

/**
 * Copy data from a buffer.
 * Depending on the gpufort::Direction parameter,
 * the source buffer is interpreted as host or device variable
 * and the destination is the array's host or device buffer.
 * \return  Error code returned by the underlying hipMemcpy operation.
 * \note Use size_in_bytes() for determining the buffer size.
 */
GPUFORT_HOST_ROUTINE gpufort::error copy_data_from_buffer{{async_suffix}}(
   void* buffer, gpufort::Direction direction{{",
   gpufort::queue stream" if is_async}}
) {
  T* dest = nullptr ;
  hipMemcpyKind memcpy_kind = hipMemcpyDefault;
  switch (direction) {
    case HostToDevice: 
      dest = this->data_dev.data;
      memcpy_kind = hipMemcpyHostToDevice;
      break;
    case DeviceToDevice:
      dest = this->data_dev.data;
      memcpy_kind = hipMemcpyDeviceToDevice;
      break;
    case HostToHost: 
      dest = this->data_host;
      memcpy_kind = hipMemcpyHostToHost
      break;
    case DeviceToHost:
      dest = this->data_host;
      memcpy_kind = hipMemcpyDeviceToHost;
      break;
    default:
      std::cerr << "ERROR: gpufort::array{{rank}}::copy_data_from_buffer{{async_suffix}}(...): Unexpected value for 'direction': " 
                << static_cast<int>(direction) << std::endl; 
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
    this->size_in_bytes(), 
    direction{{", stream" if is_async}});
}

/**
 * Copy device data to the host.
 * \return  Error code returned by the underlying hipMemcpy operation.
 */
GPUFORT_HOST_ROUTINE gpufort::error copy_data_to_host{{async_suffix}}({{"gpufort::queue stream" if is_async}}) {
  #ifndef __HIP_DEVICE_COMPILE__
  assert(this->data_host!=nullptr);
  assert(this->data_dev.data!=nullptr);
  #endif
  return hipMemcpy{{"Async" if is_async}}(
    (void*) this->data_host, 
    (void*) this->data_dev.data,
    this->size_in_bytes(), 
    hipMemcpyDeviceToHost{{", stream" if is_async}});
}

/**
 * Copy host data to the device.
 * \return  Error code returned by the underlying hipMemcpy operation.
 */
GPUFORT_HOST_ROUTINE gpufort::error copy_data_to_device{{async_suffix}}({{"gpufort::queue stream" if is_async}}) {
  #ifndef __HIP_DEVICE_COMPILE__
  assert(this->data_host!=nullptr);
  assert(this->data_dev.data!=nullptr);
  #endif
  return hipMemcpy{{"Async" if is_async}}(
    (void*) this->data_dev.data, 
    (void*) this->data_host,
    this->size_in_bytes(),
    hipMemcpyHostToDevice{{", stream" if is_async}});
}

/** 
 * Copies the struct to the device into an pre-allocated
 * device memory block but does not touch the data associated with 
 * data_dev & data_host (shallow copy).
 * \param[in] device_struct device memory address to copy to
 */
GPUFORT_HOST_ROUTINE gpufort::error copy_self_to_device{{async_suffix}}(
    gpufort::array{{rank}}<T>* device_struct{{",
    gpufort::queue stream" if is_async}}
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
GPUFORT_HOST_ROUTINE gpufort::error create_device_copy{{async_suffix}}(
    gpufort::array{{rank}}<T>** device_copy{{",
    gpufort::queue stream" if is_async}}
) {
  const size_t size = sizeof(array{{rank}}<char>); // sizeof(T*) = sizeof(char*)
  gpufort::error ierr   = hipMalloc((void**)device_copy,size);      
  if ( ierr == gpufort::success ) {
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
     * @see size_in_bytes()
     * \param[inout] pointer to the buffer to allocate
{% if is_host %}
     * \param[in] pinned If the memory should be pinned (default=True)
     * \param[in] flags  Flags for the host memory allocation (default=0).
{% endif %}
     */
    GPUFORT_HOST_ROUTINE gpufort::error allocate_{{target}}_buffer(void** buffer{{",bool pinned=true,int flags=0" if is_host}}) {
      return this->data_dev.allocate_{{target}}_buffer(buffer{{",pinned,flags" if is_host}});
    }

    /**
     * Deallocate a {{target}} buffer
     * created via the allocate_{{target}}_buffer routine.
     * @see size_in_bytes(), allocate_{{target}}_buffer
     * \param[inout] the buffer to deallocte
{% if is_host %}
     * \param[in] pinned If the memory to deallocate is pinned (default=True)
{% endif %}
     */
    GPUFORT_HOST_ROUTINE gpufort::error deallocate_{{target}}_buffer(void* buffer{{",bool pinned=true" if is_host}}) {
      return this->data_dev.deallocate_{{target}}_buffer(buffer{{",pinned" if is_host}});
    }
{% endfor %}

    /** 
     * Copy metadata from another gpufort array to this gpufort array.
     * \note The allocation mode is always set to AllocMode::WrapHostWrapDevice
     *       and the sync mode is set to SyncMode::None.
     * \note No deep copy, i.e. the host and device data is not copied, just the pointers
     * to the data.
     */
    GPUFORT_HOST_ROUTINE void copy(const gpufort::array{{rank}}<T>& other) {
      other.data_host  = this->data_host;
      other.alloc_mode = this->alloc_mode;
      other.sync_mode  = this->sync_mode;
      other.num_refs   = this->num_refs;
      other.data_dev.
      this->init(
        other.bytes_per_element,
        other.data_host,
        other.data_dev.data,
        other.size(1),{% for d in range(2,rank_ub) %}other.size({{d}}),{% endfor %}{{""}}
        other.lbound(1),{% for d in range(2,rank_ub) %}other.lbound({{d}}),{% endfor %}{{""}}
        AllocMode::WrapHostWrapDevice,
        SyncMode::None);
    }
{%- endmacro -%}
{########################################################################################}
{%- macro render_arrays(max_rank) -%}
namespace gpufort {
{% for rank in range(1,max_rank+1) %}
{% set rank_ub = rank+1 %}
  template<typename T>
  struct array{{rank}}{
    array_ptr{{rank}}<T> data_dev;
    T*                   data_host;
    AllocMode alloc_mode = AllocMode::WrapHostAllocDevice; //> Data allocation strategy. Default: 
                                                           //> wrap the host and allocate device data
    SyncMode sync_mode  = SyncMode::None;                  //> How data should be synchronized
                                                           //> during the initialization and destruction of this GPUFORT array.
    int num_refs = 0;                                      //> Number of references.

    array{{rank}}() {
      // do nothing
    }
    
    ~array{{rank}}() {
      // do nothing
    }
{% for async_suffix in ["_async",""] %}
{% set is_async = async_suffix == "_async" %}
{{ render_init_routines(rank,async_suffix) | indent(4,True) }}
{{ render_memory_operations(rank,async_suffix) | indent(4,True) }}    
{% endfor %} {# async_suffix #}
{{ render_print_routines(rank) | indent(4,True) }}
  };
{{ "" if not loop.last }}
{% endfor -%}
} // namespace gpufort

{{ gapm.render_gpufort_array_cpp_property_getters(max_rank) | indent(0,True) -}}
{%- endmacro -%}
{########################################################################################}
