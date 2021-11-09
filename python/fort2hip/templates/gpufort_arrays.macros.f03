{# SPDX-License-Identifier: MIT                                                 #}
{# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved. #}
{# Fortran side #}
{%- macro gpufort_arrays_fortran_interfaces(datatypes,max_rank) -%}
interface gpufort_mapped_array_init
{% for tuple in datatypes %}
{% for rank in range(1,max_rank+1) %}
  function gpufort_mapped_array{{rank}}_{{datatype}}_init (
      gpufort::MappedArray{{rank}}<{{datatype}}>* mapped_array,
      {{f_type}}, data_host,
      {{f_type}} data_dev,
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
end interface
 
  function gpufort_mapped_array{{rank}}_{{datatype}}_destroy(
      gpufort::MappedArray{{rank}}<{{datatype}}>* mapped_array
  ) {
    return mapped_array->destroy();
  } 
  
  function gpufort_mapped_array{{rank}}_{{datatype}}_copy_data_to_host (
      gpufort::MappedArray{{rank}}<{{datatype}}>* mapped_array
  ) {
    return mapped_array->copy_data_to_host();
  } 
  
  function gpufort_mapped_array{{rank}}_{{datatype}}_copy_data_to_device (
      gpufort::MappedArray{{rank}}<{{datatype}}>* mapped_array
  ) {
    return mapped_array->copy_data_to_device();
  } 
  
  __host__ void gpufort_mapped_array{{rank}}_{{datatype}}_inc_num_refs(
      gpufort::MappedArray{{rank}}<{{datatype}}>* mapped_array
  ) {
    mapped_array->num_refs += 1;
  } 
  
  function gpufort_mapped_array{{rank}}_{{datatype}}_dec_num_refs(
      gpufort::MappedArray{{rank}}<{{datatype}}>* mapped_array,
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
{%- endmacro -%}
