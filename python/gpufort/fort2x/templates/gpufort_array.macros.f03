{# SPDX-License-Identifier: MIT                                                 #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{# Fortran side #}
{########################################################################################}
{% import "common.macros.f03"   as cm %}
{% import "gpufort.macros.h"    as gm %}
{% import "gpufort_array_ptr.macros.h" as gapm %}
{########################################################################################}
{%- macro render_gpufort_arrays(max_rank) -%}
{########################################################################################}
{% for rank in range(1,max_rank+1) %}
! NOTE: the below types must have exactly the
! same data layout as the corresponding 
! gpufort C/C++ structs.
type,bind(c) :: gpufort_array{{rank}}
  type(gpufort_array_ptr{{rank}}) :: data_dev
  type(c_ptr) :: data_host = c_null_ptr
  integer(kind(gpufort_array_wrap_host_wrap_device)) :: alloc_mode = gpufort_array_wrap_host_alloc_device  !> Data allocation strategy. Default: 
                                                                                                           !> wrap the host and allocate device data
  integer(kind(gpufort_array_sync_none)) :: sync_mode  = gpufort_array_sync_none               !> How data should be synchronized
                                                                                               !> during the initialization and destruction of this GPUFORT array.
end type
{% endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_interfaces(datatypes,max_rank) -%}
{########################################################################################}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_data_access_interfaces(datatypes,max_rank) -%}
{########################################################################################}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_init_routines(prefix,datatypes,max_rank) -%}
{########################################################################################}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_wrap_routines(datatypes,max_rank) -%}
{########################################################################################}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_data_access_routines(prefix,datatypes,max_rank) -%}
{########################################################################################}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_allocate_buffer_routines(prefix,datatypes,max_rank) -%}
{########################################################################################}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_deallocate_buffer_routines(prefix,datatypes,max_rank) -%}
{########################################################################################}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_copy_to_from_buffer_routines(prefix,datatypes,max_rank) -%}
{########################################################################################}
{%- endmacro -%}
{########################################################################################}
