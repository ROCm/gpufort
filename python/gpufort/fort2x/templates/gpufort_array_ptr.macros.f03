{# SPDX-License-Identifier: MIT                                              #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{########################################################################################}
{% import "gpufort.macros.h" as gm %}
{########################################################################################}
{%- macro render_gpufort_array_ptrs(max_rank) -%}
{% for rank in range(1,max_rank+1) %}
{% set rank_ub = rank+1 %}
!>

!> Describes meta information of a multi-dimensional array
!> such as the lower and upper bounds per dimension, the extent per dimension, the
!> strides per dimension and the total number of elements.
type,bind(c) :: gpufort_array_ptr{{rank}}
  type(c_ptr) :: data = c_null_ptr
  type(gpufort_dope) :: dope
  integer(c_int) :: bytes_per_element = -1       !> Offset for index calculation scalar product of negative lower bounds and strides.
{% endfor %}
{%- endmacro -%}
