{# SPDX-License-Identifier: MIT                                              #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{########################################################################################}
{% import "gpufort.macros.h" as gm %}
{########################################################################################}
{%- macro render_init_routines_that_delegate_to_dope(thistype,rank,have_data_ptr=False,prefix="dope") -%}
{# prefix - gpufort::dope prefix name #}
/**
 * Init the dope vector of this {{thistype}}. 
 * \note Assigning initial data is target specific and thus not handled by this
         constructor.
 * \param[in] bytes_per_element Bytes per element of this type. The array pointer
              might have been created in Fortran where a dummy type must be used
              as there is only limited support for parameterized types in Fortran. 
 * \param[in] n1,n2,... element counts per dimension of the multi-dim. array 
 * \param[in] lb1,lb2,... lower bounds, i.e. the index of the first element per dimension of the multi-dim. array 
 */ 
GPUFORT_DEVICE_ROUTINE_INLINE void init(
{% if have_data_ptr -%}  int bytes_per_element,
  T* data,{%- endif %}
{{ gm.separated_list("const int n",",",rank) | indent(2,True) }},
{{ gm.separated_list("const int lb",",",rank) | indent(2,True) }}
) {
  this->{{prefix}}.init(
{# not so sure about this #}
/*{% if have_data_ptr -%}    bytes_per_element,data,{%- endif %}*/
{{ gm.separated_list_single_line("n",",",rank) | indent(4,True) }},
{{ gm.separated_list_single_line("lb",",",rank) | indent(4,True) }}
  );
}

/**
 * Init the dope vector of this {{thistype}}. 
 * \note Assigning initial data is target specific and thus not handled by this
         constructor.
 * \param[in] bytes_per_element Bytes per element of this type. The array pointer
              might have been created in Fortran where a dummy type must be used
              as there is only limited support for parameterized types in Fortran. 
 * \param[in] sizes extent per dimension; array with `rank` elements.  
 * \param[in] lbounds lower bound per dimension; array with `rank` elements.
 */ 
GPUFORT_DEVICE_ROUTINE_INLINE void init(
{% if have_data_ptr -%}  int bytes_per_element,
  T* data,{%- endif %}
  const int* const sizes,
  const int* const lbounds
) {
  this->init(
{% if have_data_ptr -%}    bytes_per_element,data,{%- endif %}
{{ gm.separated_list_single_line("sizes[",",",rank,"]") | indent(4,True) }},
{{ gm.separated_list_single_line("lbounds[",",",rank,"]") | indent(4,True) }}
  );
}

/** (Shallow) copy operation. Does not copy array data as this is a target specific operation. */
GPUFORT_DEVICE_ROUTINE_INLINE void copy(const {{thistype}}{{rank}}& other) {
{% if have_data_ptr %}
  this->data = other.data;
{% endif %}
  this->{{prefix}}.copy(other.{{prefix}});
}

/**
 * Init the dope vector of this {{thistype}}. 
 * \note Assigning initial data is target specific and thus not handled by this
         constructor.
 * \param[in] bytes_per_element Bytes per element of this type. The array pointer
              might have been created in Fortran where a dummy type must be used
              as there is only limited support for parameterized types in Fortran. 
 * \param[in] n1,n2,... element counts per dimension of the multi-dim. array 
 * \param[in] lb1,lb2,... lower bounds, i.e. the index of the first element per dimension of the multi-dim. array 
 */ 
GPUFORT_DEVICE_ROUTINE {{thistype}}{{rank}}(
{% if have_data_ptr -%}  T* data{%- endif %},
{{ gm.separated_list("const int n",",",rank) | indent(2,True) }},
{{ gm.separated_list("const int lb",",",rank) | indent(2,True) }}
) {
  this->init(
{% if have_data_ptr -%}    data, sizeof(T),{%- endif %}
{{ gm.separated_list_single_line("n",",",rank) | indent(4,True) }},
{{ gm.separated_list_single_line("lb",",",rank) | indent(4,True) }}
  ); 
} 

/** 
 * Init the dope vector of this {{thistype}}. 
 * \note Assigning initial data is target specific and thus not handled by this
        constructor.
 * \param[in] bytes_per_element Bytes per element of this type. The array pointer
              might have been created in Fortran where a dummy type must be used
              as there is only limited support for parameterized types in Fortran. 
 * \param[in] sizes extent per dimension; array with `rank` elements.  
 * \param[in] lbounds lower bound per dimension; array with `rank` elements.
 */
GPUFORT_DEVICE_ROUTINE_INLINE {{thistype}}{{rank}}(
{% if have_data_ptr -%}  T* data,{%- endif %}
  const int* const sizes,
  const int* const lbounds
) {
  this->init({% if have_data_ptr -%}data,sizeof(T),{%- endif %}sizes,lbounds);
} 

/** (Shallow) copy operation. Does not copy array data as this is a target specific operation. */
GPUFORT_DEVICE_ROUTINE_INLINE {{thistype}}{{rank}}(const {{thistype}}{{rank}}& other) {
  this->copy(other);
}
 
/** Default constructor. */ 
GPUFORT_DEVICE_ROUTINE_INLINE {{thistype}}{{rank}}(
) {}
{%- endmacro -%}
{########################################################################################}
{%- macro render_subarray_routines(thistype,resulttype,rank,prefix="dope") -%}
{# prefix - gpufort::dope prefix name #}
/**
 * Reset the last dimension's upper bound and adjust
 * the number of elements accordingly.
 * \param[in] ub{{rank}} new upper bound for the last dimension (dimension with largest stride).
 */
GPUFORT_DEVICE_ROUTINE_INLINE void set_last_dim_ubound(const int ub{{rank}}) {
{% if resulttype == "array" %}
{% set curr_array_ptr = "this->data_dev." %}
{% else %}
{% set curr_array_ptr = "this->" %}
{% endif %}
  const int stride{{rank}}  = {{curr_array_ptr}}{{prefix}}.stride{{rank}};
  const int lb{{rank}}      = {{curr_array_ptr}}{{prefix}}.lbound({{rank}});
  {{curr_array_ptr}}dope.num_elements = ub{{rank}} - lb{{rank}} + 1;
}

{% for d in range(rank-1,0,-1) %}
/**
 * \return a {{thistype}}{{d}} by fixing the {{rank-d}} last dimensions of this {{thistype}}{{rank}}.
 * \param[in] i{{d+1}},...,i{{rank}} indices to fix.
 */
GPUFORT_DEVICE_ROUTINE_INLINE {{resulttype}}{{d}}<T> subarray(
{% for e in range(d+1,rank+1) %}
  const int i{{e}}{{"," if not loop.last}}
{% endfor %}
) const {
{% for e in range(1,d+1) %}
  const int n{{e}}  = this->size({{e}});
  const int lb{{e}} = this->lbound({{e}});
{% endfor %}
  const int index = this->linearized_index(
{{"" | indent(8,True)}}{% for e in range(1,d+1) %}lb{{e}},{{"\n" if loop.last}}{%- endfor %}
{{"" | indent(8,True)}}{% for e in range(d+1,rank+1) %}i{{e}}{{"," if not loop.last}}{%- endfor %});
{% if resulttype == "array" %}
  T* data_host_new = nullptr; 
  T* data_dev_new  = nullptr; 
  if (this->data_host != nullptr) {
    data_host_new = this->data_host + index;
  }
  if (this->data_dev.data != nullptr) {
    data_dev_new = this->data_dev.data + index;
  }
  array{{d}}<T> result;
  result.data_host = data_host_new;
  result.alloc_mode = AllocMode::WrapHostWrapDevice;
  result.sync_mode  = SyncMode::None;
  result.data_dev.data = data_dev_new;
{% set curr_array_ptr = "this->data_dev." %}
{% set new_array_ptr  = "result.data_dev." %}
{% else %}
  T* data_new = nullptr; 
  if (this->data != nullptr) {
    data_new = &this->data[0] + index;
  }
  array_ptr{{d}}<T> result;
  result.data = data;
{% set curr_array_ptr = "this->" %}
{% set new_array_ptr  = "result." %}
{% endif %}
{% for e in range(2,d+1) %}
  {{new_array_ptr}}{{prefix}}.stride{{e}} = {{curr_array_ptr}}{{prefix}}.stride{{e}};
{% endfor %}
  {{new_array_ptr}}{{prefix}}.num_elements = {{curr_array_ptr}}{{prefix}}.stride{{d+1}};
  {{new_array_ptr}}{{prefix}}.offset       = {{curr_array_ptr}}{{prefix}}.offset % {{curr_array_ptr}}{{prefix}}.stride{{d+1}};
  return result;
}

 /**
 * \return a {{thistype}}{{d}} by fixing the {{rank-d}} last dimensions of this {{thistype}}{{rank}}.
           Further reset the upper bound of the result's last dimension.
 * \param[in] ub{{d}} upper bound for the result's last dimension.
 * \param[in] i{{d+1}},...,i{{rank}} indices to fix.
 */
GPUFORT_DEVICE_ROUTINE_INLINE {{resulttype}}{{d}}<T> subarray_w_ub(
  const int ub{{d}},
{% for e in range(d+1,rank+1) %}
  const int i{{e}}{{"," if not loop.last}}
{% endfor %}
) const {
{% if resulttype == "array" %}
  array{{d}}<T> result = 
{% else %}
  array_ptr{{d}}<T> result = 
{% endif %}  
    this->subarray(
{% for e in range(d+1,rank+1) %}
      i{{e}}{{"," if not loop.last else ");"}}
{% endfor %}
  result.set_last_dim_ubound(ub{{d}});
  return result;
}
{% endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_array_property_getters_that_delegate_to_dope(rank,fixed_size=False,prefix="dope") -%}
{# prefix - gpufort::dope prefix name #}
/**
 * Linearize multi-dimensional index.
 *
 * \param[in] i1,i2,... multi-dimensional array index.
 */
GPUFORT_DEVICE_ROUTINE_INLINE int linearized_index(
{{ gm.separated_list("const int i",",",rank) | indent(2,True) }}
) const {
  return this->{{prefix}}(
{{ gm.separated_list_single_line("i",",",rank) | indent(4,True) }}
  );
}

/**
 * \return number of array elements.
 */
GPUFORT_DEVICE_ROUTINE_INLINE int size() const {
{% if fixed_size %}
  return size;
{% else %}
  return this->{{prefix}}.size();
{% endif %}
}

/**
 * \return size of the array in dimension 'dim'.
 * \param[in] dim selected dimension: 1,...,{{rank}}
 */
GPUFORT_DEVICE_ROUTINE_INLINE int size(int dim) const {
  return this->{{prefix}}.size(dim);
}

/**
 * \return lower bound (inclusive) of the array in dimension 'dim'.
 * \param[in] dim selected dimension: 1,...,{{rank}}
 */
GPUFORT_DEVICE_ROUTINE_INLINE int lbound(int dim) const {
  return this->{{prefix}}.lbound(dim);
}

/**
 * \return upper bound (inclusive) of the array in dimension 'dim'.
 * \param[in] dim selected dimension: 1,...,{{rank}}
 */
GPUFORT_DEVICE_ROUTINE_INLINE int ubound(int dim) const {
  return this->{{prefix}}.ubound(dim);
}
{%- endmacro -%}
{########################################################################################}
{%- macro render_size_in_bytes_routines(prefix_w_dot="") -%}
{# prefix_w_dot - prefix_w_dot to derive the information from #}
/**
 * \return size of a single array element in bytes. 
 */  
GPUFORT_DEVICE_ROUTINE_INLINE int bytes_per_element_() const {
  return this->{{prefix_w_dot}}bytes_per_element;
}

/**
 * \return size of the array data in bytes. Does not include size of metadata
 * such as the dope vector.
 */
GPUFORT_DEVICE_ROUTINE_INLINE size_t size_in_bytes() const {
  return static_cast<size_t>(this->{{prefix_w_dot}}size())
         * static_cast<size_t>(this->{{prefix_w_dot}}bytes_per_element_());
}
{%- endmacro -%}
{########################################################################################}
{%- macro render_data_access_routines(rank,prefix="dope") -%}
/**
 * \return Element at given index.
 * \param[in] i1,i2,... multi-dimensional array index.
 */
GPUFORT_DEVICE_ROUTINE_INLINE T& operator() (
{{ gm.separated_list("const int i",",",rank) | indent(6,True) }}
) {
  this->{{prefix}}(
{{ gm.separated_list_single_line("i",",",rank) | indent(8,True) }}
  );
  #ifndef __HIP_DEVICE_COMPILE__
{% for r in range(1,rank+1) %}
  assert(i{{r}} >= lbound({{r}}));
  assert(i{{r}} <= ubound({{r}}));
{% endfor %}
  #endif
  return this->data[index];
}

/**
 * \return Element at given index.
 * \param[in] i1,i2,... multi-dimensional array index.
 */
GPUFORT_DEVICE_ROUTINE_INLINE const T& operator() (
{{ gm.arglist("const int i",rank) | indent(6,True) }}
) const {
  this->{{prefix}}(
{{ gm.separated_list_single_line("i",",",rank) | indent(8,True) }}
  );
  #ifndef __HIP_DEVICE_COMPILE__
{% for r in range(1,rank+1) %}
  assert(i{{r}} >= lbound({{r}}));
  assert(i{{r}} <= ubound({{r}}));
{% endfor %}
  #endif
  return this->data[index];
}

/**
 * \return Element at given index.
 * \param[in] index linear access index, starting from 0
 */
GPUFORT_DEVICE_ROUTINE_INLINE T& operator[] (
  int index
) {
  #ifndef __HIP_DEVICE_COMPILE__
  assert(index < this->size());
  #endif
  return this->data[index];
}

/**
 * \return Element at given index.
 * \param[in] index linear access index, starting from 0
 */
GPUFORT_DEVICE_ROUTINE_INLINE const T& operator[] (
  int index
) const {
  #ifndef __HIP_DEVICE_COMPILE__
  assert(index < this->size());
  #endif
  return this->data[index];
}
{%- endmacro -%}
{########################################################################################}
{%- macro render_memory_operations(gpufort_type,rank,data_member="data") -%}
{# data_member - gpufort::array_ptr data data_member expression #}
{% for target in ["host","device"] %}
{%   set is_host = target == "host" %}
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
  #ifndef __HIP_DEVICE_COMPILE__
  assert(buffer!=nullptr);
  #endif
  gpufort::error ierr = hipSuccess;
{% if target == "device" %} 
  ierr = hipMalloc(buffer,this->size_in_bytes());
{% else %}
  if ( pinned ) { 
    ierr = hipHostMalloc(buffer,this->size_in_bytes(),flags);
  } else {
    *buffer = malloc(this->size_in_bytes());
  }
{% endif %}
  return ierr;
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
  #ifndef __HIP_DEVICE_COMPILE__
  assert(buffer!=nullptr);
  #endif
  gpufort::error ierr = hipSuccess;
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
{% for async_suffix in ["_async",""] %}
{%   set is_async = async_suffix == "_async" %}
/**
 * Copy data to a buffer.
 * Depending on the direction parameter,
 * the source and destinaton buffers are interpreted as host or 
 * device variables.
 * \return  Error code returned by the memcpy operation.
 * \note Use size_in_bytes() for determining the buffer size.
 */
GPUFORT_HOST_ROUTINE gpufort::error copy_data_to_buffer{{async_suffix}}(
   void* buffer, gpufort::Direction direction{{",
   gpufort::queue stream" if is_async}}
) {
  #ifndef __HIP_DEVICE_COMPILE__
  assert(buffer!=nullptr);
  assert(this->{{data_member}}!=nullptr);
  #endif
  hipMemcpyKind memcpy_kind = hipMemcpyDefault;
  switch (direction) {
    case Direction::HostToHost: 
      memcpy_kind = hipMemcpyHostToHost;
      break;
    case Direction::HostToDevice:
      memcpy_kind = hipMemcpyHostToDevice;
      break;
    case Direction::DeviceToHost: 
      memcpy_kind = hipMemcpyDeviceToHost;
      break;
    case Direction::DeviceToDevice:
      memcpy_kind = hipMemcpyDeviceToDevice;
      break;
    default:
      std::cerr << "ERROR: gpufort::{{gpufort_type}}::copy_data_to_buffer{{async_suffix}}(...): Unexpected value for 'direction': " 
                << static_cast<int>(direction) << std::endl; 
      std::terminate();
      break;
  }
  T* src = this->{{data_member}};
  return static_cast<gpufort::error>(
    hipMemcpy{{"Async" if is_async}}(
      buffer, 
      (void*) src,
      this->size_in_bytes(), 
      memcpy_kind{{", stream" if is_async}}));
}

/**
 * Copy data from a buffer.
 * Depending on the direction parameter,
 * the source and destinaton buffers are interpreted as host or 
 * device variables.
 * \return  Error code returned by the memcpy operation.
 * \note Use size_in_bytes() for determining the buffer size.
 */
GPUFORT_HOST_ROUTINE gpufort::error copy_data_from_buffer{{async_suffix}}(
   void* buffer, gpufort::Direction direction{{",
   gpufort::queue stream" if is_async}}
) {
  hipMemcpyKind memcpy_kind = hipMemcpyDefault;
  switch (direction) {
    case Direction::HostToHost: 
      memcpy_kind = hipMemcpyHostToHost;
      break;
    case Direction::HostToDevice:
      memcpy_kind = hipMemcpyHostToDevice;
      break;
    case Direction::DeviceToHost: 
      memcpy_kind = hipMemcpyDeviceToHost;
      break;
    case Direction::DeviceToDevice:
      memcpy_kind = hipMemcpyDeviceToDevice;
      break;
    default:
      std::cerr << "ERROR: gpufort::{{gpufort_type}}::copy_data_from_buffer{{async_suffix}}(...): Unexpected value for 'direction': " 
                << static_cast<int>(direction) << std::endl; 
      std::terminate();
      break;
  }
  T* dest = this->{{data_member}};
  return static_cast<gpufort::error>(
    hipMemcpy{{"Async" if is_async}}(
      (void*) dest,
      buffer, 
      this->size_in_bytes(), 
      memcpy_kind{{", stream" if is_async}}));
}
{% endfor %}{# async_suffix #}
{%- endmacro -%}
{########################################################################################}
{%- macro render_array_property_inquiry_cpp_routines(thistype,max_rank) -%}
{% for rank in range(1,max_rank+1) %}
/**
 * \return Number of array elements.
 */
template<typename T>
GPUFORT_DEVICE_ROUTINE_INLINE int size(
    gpufort::{{thistype}}{{rank}}<T>& array) {
  return array.size();
}

/**
 * \return size of the array in dimension 'dim'.
 * \param[in] dim selected dimension: 1,...,{{rank}}
 */
template<typename T>
GPUFORT_DEVICE_ROUTINE_INLINE int size(
    gpufort::{{thistype}}{{rank}}<T>& array,
    int dim) {
  return array.size(dim);
}

/**
 * \return lower bound (inclusive) of the array in dimension 'dim'.
 * \param[in] dim selected dimension: 1,...,{{rank}}
 */
template<typename T>
GPUFORT_DEVICE_ROUTINE_INLINE int lbound(
    gpufort::{{thistype}}{{rank}}<T>& array,
    int dim) {
  return array.lbound(dim);
}

/**
 * \return upper bound (inclusive) of the array in dimension 'dim'.
 * \param[in] dim selected dimension: 1,...,{{rank}}
 */
template<typename T>
GPUFORT_DEVICE_ROUTINE_INLINE int ubound(
    gpufort::{{thistype}}{{rank}}<T>& array,
    int dim) {
  return array.ubound(dim);
}
{% endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_ptrs(max_rank) -%}
namespace gpufort {
{% for rank in range(1,max_rank+1) %}
  /**
   * Stores a pointer to an array plus a dope vector
   * that describes the layout of the array.
   * This class overloads `operator()` to retrieve elements
   * of the array according to the data layout described
   * by the dope vector. 
   */
  template<typename T>
  struct array_ptr{{rank}} {
    T* data;
    dope{{rank}} dope;
    int bytes_per_element = -1;
{{ render_init_routines_that_delegate_to_dope("array_ptr",rank,True) | indent(4,True) }} 
{{ render_array_property_getters_that_delegate_to_dope(rank) | indent(4,True) }}
{{ render_size_in_bytes_routines() | indent(4,True) }}
{{ render_data_access_routines(rank) | indent(4,True) }}
{{ render_memory_operations("array_ptr",rank,"data") }}
{{ render_subarray_routines("array_ptr","array_ptr",rank) | indent(4,true) }}
  }; // class array_ptr
{% endfor %}
{{ render_array_property_inquiry_cpp_routines("array_ptr",max_rank) | indent(2,True) -}}
} // namespace gpufort
{%- endmacro -%}
{########################################################################################}
