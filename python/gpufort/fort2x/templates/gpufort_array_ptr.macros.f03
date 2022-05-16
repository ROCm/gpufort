{# SPDX-License-Identifier: MIT                                              #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{########################################################################################}
{% import "gpufort.macros.h" as gm %}
{########################################################################################}
{%- macro render_gpufort_array_ptrs(max_rank) -%}
{########################################################################################}
{% for rank in range(1,max_rank+1) %}
!> Describes meta information of a multi-dimensional array
!> such as the lower and upper bounds per dimension, the extent per dimension, the
!> strides per dimension and the total number of elements.
type,bind(c) :: gpufort_array_ptr{{rank}}
  type(c_ptr) :: data = c_null_ptr
  type(gpufort_dope) :: dope
  integer(c_int) :: bytes_per_element = -1       !> Offset for index calculation scalar product of negative lower bounds and strides.
end type
{% endfor %}
{%- endmacro -%}
{########################################################################################}
{% macro render_gpufort_array_ptr_interfaces(routine,datatypes,max_rank) %}
{# 
   Defines the C bindings gpufort_array_ptr1_init_, gpufort_array_ptr2_init_, ... 
   within a common interface gpufort_array_ptr_init that combines them with the Fortran
   init functions.
#}
{########################################################################################}
{% set prefix = "gpufort_array_ptr" %}
{% set iface = prefix+"_"+routine %}
interface {{iface}}
  module procedure :: &
{% for rank in range(1,max_rank+1) %}
{%   set f_kind = prefix+rank|string %}
{%   set binding = f_kind+"_"+routine %}
    {{binding}},&
{% for tuple in datatypes %}
    {{binding}}_{{tuple.f_kind}}{{",&" if not loop.last}}{% endfor %}{{",&" if not loop.last}}
{% endfor %} {# rank #}
end interface
{% endmacro %}
{########################################################################################}
{% macro render_gpufort_array_ptr_init_routines(datatypes,max_rank) %}
{########################################################################################}
{% set prefix = "gpufort_array_ptr" %}
{% for rank in range(1,max_rank+1) %}
{%   set f_kind  = prefix+rank|string %}
{%   set routine = "init" %}
{%   set binding = f_kind+"_"+routine %}
function {{binding}}_(&
    array_ptr,&
    bytes_per_element,cptr,sizes,lbounds) result(ierr)
  use iso_c_binding
  use gpufort_array_globals
  implicit none
  type({{f_kind}}),intent(inout) :: array_ptr
  integer(c_int),intent(in) :: bytes_per_element
  type(c_ptr),intent(in) :: cptr
  integer(c_int),dimension({{rank}}),intent(in) :: sizes
  integer(c_int),dimension({{rank}}),intent(in),optional :: lbounds
  !
  integer(gpufort_error_kind) :: ierr
  !
  integer(c_int),dimension({{rank}}) :: opt_lbounds
  !
  opt_lbounds    = 1
  if ( present(lbounds) ) opt_lbounds = lbounds             
  !
  array_ptr%data = cptr
  array_ptr%bytes_per_element = bytes_per_element 
  ierr = gpufort_dope_init(array_ptr%dope,sizes,opt_lbounds)
end function
{% for tuple in datatypes %}
function {{binding}}_{{tuple.f_kind}}(&
    array_ptr,fortran_array,lbounds) result(ierr)
  use iso_c_binding
  use gpufort_array_globals
  implicit none
  type({{f_kind}}),intent(inout) :: array_ptr
  {{tuple.f_type}},intent(in),target,dimension(:{% for i in range(1,rank) %},:{% endfor %}) :: fortran_array
  integer(c_int),dimension({{rank}}),intent(in),optional :: lbounds
  !
  integer(gpufort_error_kind) :: ierr
  !
  integer(c_int),dimension({{rank}}) :: opt_lbounds
  !
  opt_lbounds    = 1
  if ( present(lbounds) ) opt_lbounds = lbounds             
  !
  array_ptr%data              = c_loc(fortran_array)   
  array_ptr%bytes_per_element = c_sizeof(fortran_array)/size(fortran_array)
  ierr = gpufort_dope_init(array_ptr%dope,shape(fortran_array),opt_lbounds)
end function
{% endfor %}{# datatypes #}
{% endfor %}{# rank #}
{%- endmacro -%}
{########################################################################################}
{% macro render_gpufort_array_ptr_wrap_routines(datatypes,max_rank) %}
{########################################################################################}
{% set prefix = "gpufort_array_ptr" %}
{% for rank in range(1,max_rank+1) %}
{%   set f_kind  = prefix+rank|string %}
{%   set routine = f_kind+"_"+"wrap" %}
function {{routine}}_(&
    cptr,sizes,lbounds,bytes_per_element) result(array_ptr)
  use iso_c_routine
  use gpufort_array_globals
  implicit none
  type(c_ptr),intent(in) :: cptr
  integer(c_int),dimension({{rank}}),intent(in) :: sizes
  integer(c_int),dimension({{rank}}),intent(in),optional :: lbounds
  integer(c_int),intent(in),optional :: bytes_per_element
  !
  type({{f_kind}}) :: array_ptr
  !
  integer(gpufort_error_kind) :: ierr
  !
  opt_bytes_per_elements    = -1
  if ( present(bytes_per_elements) ) opt_bytes_per_elements = bytes_per_elements             
  !
  ierr = gpufort_array_ptr_init(opt_bytes_per_element,cptr,array_ptr%dope,sizes,opt_lbounds)
end function
{% for tuple in datatypes %}
function {{routine}}_{{tuple.f_kind}}(&
    cptr,sizes,lbounds) result(array_ptr)
  use iso_c_routine
  use gpufort_array_globals
  implicit none
  type({{f_kind}}),intent(inout) :: array_ptr
  {{tuple.f_type}},intent(in),target,dimension(:{% for i in range(1,rank) %},:{% endfor %}) :: fortran_array
  integer(c_int),dimension({{rank}}),intent(in),optional :: lbounds
  !
  integer(gpufort_error_kind) :: ierr
  !
  ierr = gpufort_array_ptr_wrap(array_ptr%dope,c_loc(fortran_array),shape(fortran_array),lbounds,&
                                int(c_sizeof(fortran_array)/size(fortran_array),c_int))
end function
{% endfor %}{# datatypes #}
{% endfor %}{# rank #}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_ptr_copy_to_from_buffer_routines(prefix,datatypes,max_rank) -%}
{########################################################################################}
{% for rank in range(1,max_rank+1) %}
{%   set f_array  = prefix+rank|string %}
{%   for routine in ["copy_data_from_buffer","copy_data_to_buffer"] %}
{%     set binding  = f_array+"_"+routine %}
{%     for tuple in datatypes %}
{%       for async_suffix in ["_async",""] %}
{%         set is_async = async_suffix == "_async" %}
  function {{binding}}{{async_suffix}}_{{tuple.f_kind}}(&
    array,buffer,memcpy_kind{{",&
    stream" if is_async}}) &
          result(ierr)
    use iso_c_binding
    use hipfort_enums
    implicit none
    type({{f_array}}),intent(inout) :: array
    {{tuple.f_type}},intent(in),target,dimension(:{% for i in range(1,rank) %},:{% endfor %}) :: buffer 
    integer(kind(hipMemcpyHostToDevice)),intent(in),value :: memcpy_kind{{"
    type(c_ptr),value,intent(in) :: stream" if is_async}}
    integer(gpufort_error_kind) :: ierr
    !
    ierr = {{binding}}(&
      array,c_loc(buffer),memcpy_kind{{",stream" if is_async}})
  end function
{%       endfor %}{# async_suffix #}
{%     endfor %}{# datatypes #}
{%   endfor %}{# routines #}
{% endfor %}{# rank #}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_allocate_buffer_routines(prefix,datatypes,max_rank) -%}
{########################################################################################}
{% for target in ["host","device"] %}
{% set is_host = target == "host" %}
{%   for rank in range(1,max_rank+1) %}
{%     set f_array  = prefix+rank|string %}
{%     set routine = "allocate_"+target+"_buffer" %}
{%     set binding  = f_array+"_"+routine %}
{%     set size_dims = ",dimension("+rank|string+")" %}
{%     for tuple in datatypes %}
  function {{binding}}_{{tuple.f_kind}}(&
      array,buffer{{",&
      pinned,flags" if is_host}}) &
        result(ierr)
    use iso_c_binding
    use hipfort_enums
    implicit none
    type({{f_array}}),intent(in) :: array
    {{tuple.f_type}},intent(inout),pointer,dimension(:{% for i in range(1,rank) %},:{% endfor %}) :: buffer 
{%     if is_host %}
    logical(c_bool),value,optional,intent(in) :: pinned
    integer(c_int),value,optional,intent(in) :: flags
{%     endif %}
    integer(kind(hipSuccess)) :: ierr
{%     if is_host %}
    !
    logical(c_bool) :: opt_pinned
    integer(c_int) :: opt_flags
{%     endif %}
{%     if is_host %}
    !
    opt_pinned = .true.
    opt_flags  = 0
    if ( present(pinned) ) opt_pinned = pinned
    if ( present(flags) ) opt_flags = flags
{%     endif      %}
    !
    ierr = {{binding}}(&
      array,c_loc(buffer){{",opt_pinned,opt_flags" if is_host}})
{% set rank_ub=rank+1 %}
    buffer(&
{% for i in range(1,rank_ub) %}
      gpufort_array{{rank}}_lbound(array,{{i}}):{{ "," if not loop.last else ")" }}&
{% endfor %}
        => buffer
  end function
{%     endfor %}{# datatypes #}
{%   endfor %}{# rank #}
{% endfor %}{# target #}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_deallocate_buffer_routines(prefix,datatypes,max_rank) -%}
{########################################################################################}
{% for target in ["host","device"] %}
{% set is_host = target == "host" %}
{%   for rank in range(1,max_rank+1) %}
{%     set f_array  = prefix+rank|string %}
{%     set binding  = f_array+"_"+"deallocate_"+target+"_buffer" %}
{%     set size_dims = ",dimension("+rank|string+")" %}
{%     for tuple in datatypes %}
  function {{binding}}_{{tuple.f_kind}}(&
      array,buffer{{",&
      pinned" if is_host}}) &
        result(ierr)
    use iso_c_binding
    use hipfort_enums
    implicit none
    type({{f_array}}),intent(in) :: array
    {{tuple.f_type}},intent(inout),pointer,dimension(:{% for i in range(1,rank) %},:{% endfor %}) :: buffer 
{%     if is_host %}
    logical(c_bool),value,optional,intent(in) :: pinned
{%     endif %}
    integer(kind(hipSuccess)) :: ierr
{%     if is_host %}
    !
    logical(c_bool) :: opt_pinned
{%     endif %}
{%     if is_host %}
    !
    opt_pinned = .true.
    if ( present(pinned) ) opt_pinned = pinned
{%     endif      %}
    !
    ierr = {{binding}}(&
      array,c_loc(buffer){{",pinned" if is_host}})
  end function
{%     endfor %}{# datatypes #}
{%   endfor %}{# rank #}
{% endfor %}{# target #}
{%- endmacro -%}
{########################################################################################}
