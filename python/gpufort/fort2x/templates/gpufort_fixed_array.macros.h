{# SPDX-License-Identifier: MIT                                              #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{########################################################################################}
{% import "gpufort_array_ptr.macros.h" as gapm %}
{########################################################################################}
{%- macro render_gpufort_fixed_arrays(max_rank) -%}
namespace gpufort {
{% for rank in range(1,max_rank+1) %}
{% set rank_ub = rank+1 %}
  /** 
   * Models fixed size multi-dimensional array with
   * non-trivial lower bounds.
   * \tparam T array element type
   * \tparma size number of array elements
   */
  namespace <typename T, int size>
  struct fixed_array{{rank}} {
    T data[size];
    dope{{rank}} dope;

{{ gapm.render_init_routines_that_delegate_to_dope("fixed_array",rank) | indent(4,True) }} 
{{ gapm.render_array_property_getters_that_delegate_to_dope(rank,fixed_size=True) | indent(4,True) }}
    /**
     * \return size of the array data in bytes. Does not include size of metadata
     * such as the dope vector.
     */
    GPUFORT_DEVICE_ROUTINE_INLINE size_t size_in_bytes() const {
      return sizeof(data);
    }

    /**
     * \return size of a single array element in bytes. 
     */  
    GPUFORT_DEVICE_ROUTINE_INLINE int bytes_per_element() const {
      return sizeof(T);
    }
{{ gapm.render_data_access_routines(rank) | indent(4,True) }}
{{ gapm.render_memory_operations(rank) | indent(4,True) }}
{{ gapm.render_subarray_routines("fixed_array","array_ptr",rank) | indent(4,True) }}
  }; // class fixed_array
{% endfor %}

{{ gapm.render_array_property_inquiry_cpp_routines("array_ptr",max_rank) | indent(2,True) -}}
} // namespace gpufort
{%- endmacro -%}
{########################################################################################}
