{# SPDX-License-Identifier: MIT                                                 #}
{# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved. #}
{# requires that the bound_args and bound_args_single_line macros exist #}
{% import "templates/gpufort.macros.h" as gm %}
{%- macro gpufort_arrays_c_bindings(datatypes,max_rank) -%}
{# C side #}
extern "C" {
{% for rank in range(1,max_rank+1) %}
{% set c_type = "char" %}
{% set c_prefix = "gpufort_mapped_array_"+rank|string %}
  __host__ hipError_t {{c_prefix}}_init (
      gpufort::MappedArray{{rank}}<{{c_type}}>* mapped_array,
      int bytes_per_element,
      {{c_type}}* data_host,
      {{c_type}}* data_dev,
      int* sizes,
      int* lower_bounds,
      bool pinned,
      bool copyout_at_destruction
  ) {
    return mapped_array->init(
      bytes_per_element,
      data_host, data_dev,
      sizes, lower_bounds,
      pinned, copyout_at_destruction
    );
  } 
  
  __host__ hipError_t {{c_prefix}}_destroy(
      gpufort::MappedArray{{rank}}<{{c_type}}>* mapped_array,
      hipStream_t stream = nullptr
  ) {
    return mapped_array->destroy(stream);
  } 
  
  __host__ hipError_t {{c_prefix}}_copy_data_to_host (
      gpufort::MappedArray{{rank}}<{{c_type}}>* mapped_array,
      hipStream_t stream = nullptr
  ) {
    return mapped_array->copy_data_to_host(stream);
  } 
  
  __host__ hipError_t {{c_prefix}}_copy_data_to_device (
      gpufort::MappedArray{{rank}}<{{c_type}}>* mapped_array,
      hipStream_t stream = nullptr
  ) {
    return mapped_array->copy_data_to_device(stream);
  } 
  
  __host__ void {{c_prefix}}_inc_num_refs(
      gpufort::MappedArray{{rank}}<{{c_type}}>* mapped_array
  ) {
    mapped_array->num_refs += 1;
  } 
  
  __host__ hipError_t {{c_prefix}}_dec_num_refs(
      gpufort::MappedArray{{rank}}<{{c_type}}>* mapped_array,
      bool destroy_if_zero_refs,
      hipStream_t stream = nullptr
  ) {
    mapped_array->num_refs -= 1;
    if ( destroy_if_zero_refs && mapped_array->num_refs == 0 ) {
      return mapped_array->destroy(stream);
    } {
      return hipSuccess;
    }
  } 
{% endfor %}
} // extern "C"
{%- endmacro -%}
