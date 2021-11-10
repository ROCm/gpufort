{# SPDX-License-Identifier: MIT                                                 #}
{# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved. #}
{# requires that the bound_args and bound_args_single_line macros exist #}
{% import "templates/gpufort.macros.h" as gm %}
{%- macro gpufort_arrays_c_bindings(datatypes,max_rank) -%}
{# C side #}
extern "C" {
{% for tuple in datatypes %}
{% set c_type = tuple.c_type %}
{% for rank in range(1,max_rank+1) %}
{% set c_prefix = "gpufort_mapped_array_"+rank|string+"_"+c_type %}
  __host__ hipError_t {{c_prefix}}_init (
      gpufort::MappedArray{{rank}}<{{c_type}}>* mapped_array,
      {{c_type}}* data_host,
      {{c_type}}* data_dev,
{{ gm.bound_args("const int ",rank) | indent(6,True) }},
      bool pinned,
      hipStream_t stream,
      bool copyout_at_destruction
  ) {
    return mapped_array->init(
      data_host, data_dev,
{{ gm.bound_args_single_line("",rank) | indent(6,True) }},
      pinned, stream, copyout_at_destruction
    );
  } 
  
  __host__ hipError_t {{c_prefix}}_destroy(
      gpufort::MappedArray{{rank}}<{{c_type}}>* mapped_array
  ) {
    return mapped_array->destroy();
  } 
  
  __host__ hipError_t {{c_prefix}}_copy_data_to_host (
      gpufort::MappedArray{{rank}}<{{c_type}}>* mapped_array
  ) {
    return mapped_array->copy_data_to_host();
  } 
  
  __host__ hipError_t {{c_prefix}}_copy_data_to_device (
      gpufort::MappedArray{{rank}}<{{c_type}}>* mapped_array
  ) {
    return mapped_array->copy_data_to_device();
  } 
  
  __host__ void {{c_prefix}}_inc_num_refs(
      gpufort::MappedArray{{rank}}<{{c_type}}>* mapped_array
  ) {
    mapped_array->num_refs += 1;
  } 
  
  __host__ hipError_t {{c_prefix}}_dec_num_refs(
      gpufort::MappedArray{{rank}}<{{c_type}}>* mapped_array,
      bool destroy_if_zero_refs
  ) {
    mapped_array->num_refs -= 1;
    if ( destroy_if_zero_refs && mapped_array->num_refs == 0 ) {
      return mapped_array->destroy();
    } {
      return hipSuccess;
    }
  } 
{% endfor %}
{% endfor %}
}
{%- endmacro -%}
