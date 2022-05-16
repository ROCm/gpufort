{# SPDX-License-Identifier: MIT                                              #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{########################################################################################}
{% import "gpufort.macros.h" as gm %}
{########################################################################################}
{%- macro render_gpufort_dopes(max_rank) -%}
{########################################################################################}
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
{########################################################################################}
{% macro render_gpufort_dope_array_property_inquiry_interfaces(array_type,max_rank) %}
{# 
   Defines the C bindings gpufort_dope1_size, gpufort_dope1_lbound and so on
   and puts them into the interfaces
   gpufort_dope_size, gpufort_dope_lbound, and gpufort_dope_ubound.
#}
{########################################################################################}
{% for routine in ["size","lbound","ubound"] %}
{%   set iface = array_type+"_"+routine %}
interface {{iface}}
{%   for rank in range(1,max_rank+1) %}
{%     set f_kind = array_type+rank|string %}
{%     set binding = f_kind+"_"+routine %}
  function {{binding}}(array,d) &
        bind(c,name="{{binding}}") &
          result(retval)
    use iso_c_binding
    import {{f_kind}}
    implicit none
    type({{f_kind}}),intent(in) :: array
    integer(c_int),intent(in),value :: d
    !
    integer(c_int) :: retval
  end function
{%   endfor %}{# rank #}
end interface
{% endfor %}{# routine #}
{% endmacro %}
{########################################################################################}
{% macro render_gpufort_dope_init_interfaces(max_rank) %}
{# 
   Defines the C bindings gpufort_dope1_init_, gpufort_dope2_init_, ... 
   within a common interface gpufort_dope_init that combines them with the Fortran
   init functions.
#}
{########################################################################################}
{% set prefix = "gpufort_dope" %}
{% set routine = "init" %}
{% set iface = prefix+"_"+routine %}
interface {{iface}}
{% for rank in range(1,max_rank+1) %}
{%   set f_kind  = prefix+rank|string %}
{%   set binding  = f_kind+"_"+routine %}
  function {{binding}}_(&
      array, sizes, lbounds) &
        bind(c,name="{{binding}}") &
          result(ierr)
    use iso_c_binding
    use hipfort_enums
    import {{f_kind}}
    implicit none
    type({{f_kind}}),intent(inout) :: array
    integer(c_int),dimension({{rank}}),intent(in) :: sizes, lbounds
    !
    integer(gpufort_error_kind) :: ierr
  end function
{%   endfor %}
  module procedure :: &
{% for rank in range(1,max_rank+1) %}
{%   set f_kind  = prefix+rank|string %}
{%   set binding  = f_kind+"_"+routine %}
    {{binding}}{{",&" if not loop.last}}
{% endfor %} {# rank #}
end interface
{% endmacro %}
{########################################################################################}
{% macro render_gpufort_dope_init_routines(max_rank) %}
{# 
   Defines the routines gpufort_dope1_init, gpufor_dope2_init, ...
   that call the gpufort_dope1_init_, gpufort_dope2_init_, ... C bindings.
#}
{########################################################################################}
{% set prefix = "gpufort_dope" %}
{% for rank in range(1,max_rank+1) %}
{%   set f_kind  = prefix+rank|string %} {# gpufort_dope1 #}
{%   set routine = "init" %}
{%   set binding  = f_kind+"_"+routine %} {# gpufort_dope1_init #}
function {{binding}}(&
    array,sizes,lbounds) result(ierr)
  use iso_c_binding
  use gpufort_array_globals
  implicit none
  type({{f_kind}}),intent(inout) :: array
  integer(c_int),dimension({{rank}}),intent(in) :: sizes
  integer(c_int),dimension({{rank}}),intent(in),optional :: lbounds
  !
  integer(gpufort_error_kind) :: ierr
  !
  integer(c_int),dimension({{rank}}) :: opt_lbounds
  !
  opt_lbounds    = 1
  if ( present(lbounds) ) opt_lbounds    = lbounds             
  ierr = {{binding}}_(&
    array,sizes, opt_lbounds)
end function
{% endfor %} {# rank #}
{% endmacro %}
{########################################################################################}
