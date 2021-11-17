{# SPDX-License-Identifier: MIT                                                 #}
{# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved. #}
{# requires that the bound_args and bound_args_single_line macros exist #}
{% import "templates/gpufort.macros.h" as gm %}
{%- macro gpufort_array_c_bindings(datatypes,max_rank) -%}
{# C side #}
extern "C" {
{% for rank in range(1,max_rank+1) %}
{% set c_type = "char" %}
{% set c_prefix = "gpufort_array"+rank|string %}
  __host__ hipError_t {{c_prefix}}_init (
      gpufort::array{{rank}}<{{c_type}}>* array,
      int bytes_per_element,
      {{c_type}}* data_host,
      {{c_type}}* data_dev,
      int* sizes,
      int* lower_bounds,
      gpufort::AllocMode alloc_mode,
      gpufort::SyncMode sync_mode,
      hipStream_t stream
  ) {
    return array->init(
      bytes_per_element,
      data_host, data_dev,
      sizes, lower_bounds,
      alloc_mode, sync_mode, stream
    );
  } 
  
  __host__ hipError_t {{c_prefix}}_destroy(
      gpufort::array{{rank}}<{{c_type}}>* array,
      hipStream_t stream = nullptr
  ) {
    return array->destroy(stream);
  } 
  
  __host__ hipError_t {{c_prefix}}_copy_to_host (
      gpufort::array{{rank}}<{{c_type}}>* array,
      hipStream_t stream = nullptr
  ) {
    return array->copy_to_host(stream);
  } 
  
  __host__ hipError_t {{c_prefix}}_copy_to_device (
      gpufort::array{{rank}}<{{c_type}}>* array,
      hipStream_t stream = nullptr
  ) {
    return array->copy_to_device(stream);
  } 
  
  __host__ void {{c_prefix}}_inc_num_refs(
      gpufort::array{{rank}}<{{c_type}}>* array
  ) {
    array->num_refs += 1;
  } 
  
  __host__ hipError_t {{c_prefix}}_dec_num_refs(
      gpufort::array{{rank}}<{{c_type}}>* array,
      bool destroy_if_zero_refs,
      hipStream_t stream = nullptr
  ) {
    array->num_refs -= 1;
    if ( destroy_if_zero_refs && array->num_refs == 0 ) {
      return array->destroy(stream);
    } {
      return hipSuccess;
    }
  } 
{% endfor %}
} // extern "C"
{%- endmacro -%}