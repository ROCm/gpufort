{# SPDX-License-Identifier: MIT                                              #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{########################################################################################}
{% import "gpufort.macros.h" as gm %}
{########################################################################################}
{%- macro render_gpufort_dopes(max_rank) -%}
{% for rank in range(1,max_rank+1) %}
{% set rank_ub = rank+1 %}
!> Describes meta information of a multi-dimensional array
!> such as the lower and upper bounds per dimension, the extent per dimension, the
!> strides per dimension and the total number of elements.
type,bind(c) :: gpufort_dope{{rank}}
  integer(c_int) :: index_offset = -1       !> Offset for index calculation scalar product of negative lower bounds and strides.
{% for d in range(2,rank_ub) %}
  integer(c_int) :: stride{{d}} = -1        !> Stride {{d}} for linearizing {{rank}}-dimensional index.
{% endfor %}
  integer(c_int) :: num_elements = 0        !> Number of elements represented by this array.
end type ! gpufort_dope{{rank}}
{% endfor %}
{%- endmacro -%}
