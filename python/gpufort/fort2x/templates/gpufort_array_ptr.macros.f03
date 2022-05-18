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
  integer(c_int) :: bytes_per_element = -1
end type
{% endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_module_procedure(gpufort_type,
                                  max_rank,
                                  routine,
                                  datatypes) -%}
{########################################################################################}
module procedure :: &{{"\n"}}
{%- for rank in range(1,max_rank+1) -%}
{%-   set size_dims = ",dimension("+rank|string+")" -%}
{%-   set f_kind  = gpufort_type+rank|string -%}
{%-   set binding  = f_kind+"_"+routine -%}
{%-   for tuple in datatypes -%}
{{binding | indent(4,True)}}_{{tuple.f_kind}}{{",&\n" if not loop.last}}
{%-   endfor -%}{{",&\n" if not loop.last}}
{%- endfor %}{{""}}
{%- endmacro -%}
{########################################################################################}
{% macro render_gpufort_array_ptr_interfaces(routine,datatypes,max_rank) %}
{# 
   Defines the C bindings gpufort_array_ptr1_init_, gpufort_array_ptr2_init_, ... 
   within a common interface gpufort_array_ptr_init that combines them with the Fortran
   init functions.
#}
{########################################################################################}
{% set gpufort_type = "gpufort_array_ptr" %}
{% set iface = gpufort_type+"_"+routine %}
interface {{iface}}
  module procedure :: &
{% for rank in range(1,max_rank+1) %}
{%   set f_kind = gpufort_type+rank|string %}
{%   set binding = f_kind+"_"+routine %}
    {{binding}}_,&
{% for tuple in datatypes %}
    {{binding}}_{{tuple.f_kind}}{{",&" if not loop.last}}{% endfor %}{{",&" if not loop.last}}
{% endfor %} {# rank #}
end interface
{% endmacro %}
{########################################################################################}
{% macro render_gpufort_array_ptr_init_routines(datatypes,max_rank) %}
{########################################################################################}
{% set gpufort_type = "gpufort_array_ptr" %}
{% for rank in range(1,max_rank+1) %}
{%   set f_kind  = gpufort_type+rank|string %}
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
{% set gpufort_type = "gpufort_array_ptr" %}
{% for rank in range(1,max_rank+1) %}
{%   set f_kind  = gpufort_type+rank|string %}
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
{%- macro render_gpufort_array_ptr_subarray_interfaces(gpufort_type,gpufort_subarray_type,max_rank) -%}
{# SUBARRAY #}
{########################################################################################}
{% for routine in ["subarray","subarray_w_bounds"] %}
{%   set iface = gpufort_type+"_"+routine %}
interface {{iface}}
{%   for rank in range(1,max_rank+1) %}
{%     set f_kind  = gpufort_type+rank|string %}
{%     set binding  = f_kind+"_"+routine %}
{%     for d in range(rank-1,0,-1) %}
  !> \return a {{gpufort_type}}{{d}} by fixing the {{rank-d}} last dimensions of this {{gpufort_type}}{{rank}}.
  {%   if routine == "subarray_w_ub" %}
  !> Further reset the upper bound of the result's last dimension.
  !> \param[in] ub{{d}} upper bound for the result's last dimension.
{%   endif %}
  !> \param[in] i{{d+1}},...,i{{rank}} indices to fix.
{%       set f_kind_subarray  = gpufort_subarray_type+d|string %}
  subroutine {{binding}}_{{d}} (subarray,array,&
{%       if routine == "subarray_w_ub" %}
      u{{d}},&
{%       endif %}
{{"" | indent(6,True)}}{% for e in range(d+1,rank+1) %}i{{e}}{{"," if not loop.last}}{%- endfor %}) &
        bind(c,name="{{binding}}_{{d}}")
    use iso_c_binding
    use hipfort_enums
    import {{f_kind}}
    import {{f_kind_subarray}}
    implicit none
    type({{f_kind_subarray}}),intent(inout) :: subarray
    type({{f_kind}}),intent(in) :: array
    integer(c_int),value,intent(in) :: &
{%       if routine == "subarray_w_ub" %}
      u{{d}},&
{%       endif %}
{{"" | indent(6,True)}}{% for e in range(d+1,rank+1) %}i{{e}}{{"," if not loop.last else "\n"}}{%- endfor %}
  end subroutine
{%     endfor %}{# d #}
{%   endfor %}{# rank #}
{% endfor %}{# routine #}
end interface
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_ptr_copy_data_to_from_buffer_interfaces(gpufort_type,datatypes,max_rank) -%}
{########################################################################################}
{% for async_suffix in ["_async",""] %}
{%   set is_async = async_suffix == "_async" %}
{# COPY_FROM_BUFFER, COPY_TO_BUFFER #}
{%   for routine in ["copy_data_from_buffer","copy_data_to_buffer"] %}
{%     set iface = gpufort_type+"_"+routine %}
!>
{%     if routine == "copy_data_to_buffer" %}
!> Copy this {{gpufort_type}}'s host or device data TO a host or device buffer.
{%     else %}
!> Copy this {{gpufort_type}}'s host or device data FROM a host or device buffer.
{%     endif %}
!>
interface {{iface}}{{async_suffix}}
{%     for rank in range(1,max_rank+1) %}
{%       set f_kind  = gpufort_type+rank|string %}
{%       set binding  = f_kind+"_"+routine %}
  function {{binding}}{{"_" if async_suffix == "" else async_suffix}}(&
    array,buffer,memcpy_kind{{",&
    stream" if is_async}}) &
        bind(c,name="{{binding}}{{async_suffix}}") &
          result(ierr)
    use iso_c_binding
    use gpufort_array_globals
    import {{f_kind}}
    implicit none
    type({{f_kind}}),intent(inout) :: array
    type(c_ptr),value,intent(in) :: buffer
    integer(kind(gpufort_host_to_host)),intent(in),value :: memcpy_kind{{"
    type(c_ptr),value,intent(in) :: stream" if is_async}}
    integer(gpufort_error_kind) :: ierr
  end function
{%     endfor %}
{{ render_module_procedure(gpufort_type,
                           max_rank+1,
                           routine,
                           datatypes) | indent(2,True) }}
end interface
{%   endfor %}
{% endfor %}{# async_suffix #}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_ptr_allocate_interface(gpufort_type,datatypes,max_rank) -%}
{########################################################################################}
{# ALLOCATE_BUFFER #}
{% for target in ["host","device"] %}
{%   set is_host = target == "host" %}
{%   set routine = "allocate_"+target+"_buffer" %}
{%   set iface = gpufort_type+"_"+routine %}
!> Allocate a {{target}} buffer with
!> the same size as the data buffers
!> associated with this gpufort array.
!> @see size_in_bytes()
!> \param[inout] pointer to the buffer to allocate
{%   if is_host %}
!> \param[in] pinned If the memory should be pinned.
!> \param[in] flags  Flags to pass to host memory allocation.
{%   endif %}
interface {{iface}}
{%   for rank in range(1,max_rank+1) %}
{%     set f_kind  = gpufort_type+rank|string %}
{%     set binding  = f_kind+"_"+routine %}
  function {{binding}}(&
      array,buffer{{",pinned,flags" if is_host}}) &
        bind(c,name="{{binding}}") &
          result(ierr)
    use iso_c_binding
    use gpufort_array_globals
    import {{f_kind}}
    implicit none
    type({{f_kind}}),intent(in) :: array
    type(c_ptr),intent(in) :: buffer
{%     if is_host %}
    logical(c_bool),value :: pinned
    integer(c_int),value :: flags
{%     endif %}
    integer(gpufort_error_kind) :: ierr
  end function
{%   endfor %}
{{ render_module_procedure(gpufort_type,
                           max_rank+1,
                           routine,
                           datatypes) | indent(2,True) }}
end interface
{% endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_ptr_deallocate_interface(gpufort_type,datatypes,max_rank) -%}
{# DEALLOCATE_BUFFER #}
{########################################################################################}
{% for target in ["host","device"] %}
{%   set is_host = target == "host" %}
{%   set routine = "deallocate_"+target+"_buffer" %}
{%   set iface = gpufort_type+"_"+routine %}
!> Deallocate a {{target}} buffer
!> created via the allocate_{{target}}_buffer routine.
!> \see size_in_bytes(), allocate_{{target}}_buffer
!> \param[inout] the buffer to deallocate
{%   if is_host %}
!> \param[in] pinned If the memory to deallocate is pinned.
{%   endif %}
interface {{iface}}
{%   for rank in range(1,max_rank+1) %}
{%     set f_kind  = gpufort_type+rank|string %}
{%     set binding  = f_kind+"_"+routine %}
  function {{binding}}_(&
      array,buffer{{",pinned" if is_host}}) &
        bind(c,name="{{binding}}") &
          result(ierr)
    use iso_c_binding
    use gpufort_array_globals
    import {{f_kind}}
    implicit none
    type({{f_kind}}),intent(in) :: {{gpufort_type}}
    type(c_ptr),value,intent(in) :: buffer
{%     if is_host %}
    logical(c_bool),value :: pinned
{%     endif %}
    integer(gpufort_error_kind) :: ierr
  end function
{%   endfor %}
{{ render_module_procedure(gpufort_type,
                           max_rank+1,
                           routine,
                           datatypes) | indent(2,True) }}
end interface
{% endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_ptr_copy_to_from_buffer_routines(gpufort_type,datatypes,max_rank) -%}
{########################################################################################}
{% for rank in range(1,max_rank+1) %}
{%   set f_kind  = gpufort_type+rank|string %}
{%   for routine in ["copy_data_from_buffer","copy_data_to_buffer"] %}
{%     set binding  = f_kind+"_"+routine %}
{%     for tuple in datatypes %}
{%       for async_suffix in ["_async",""] %}
{%         set is_async = async_suffix == "_async" %}
function {{binding}}{{async_suffix}}_{{tuple.f_kind}}(&
  array,buffer,memcpy_kind{{",&
  stream" if is_async}}) &
        result(ierr)
  use iso_c_binding
  use gpufort_array_globals
  implicit none
  type({{f_kind}}),intent(inout) :: array
  {{tuple.f_type}},intent(in),target,dimension(:{% for i in range(1,rank) %},:{% endfor %}) :: buffer 
  integer(kind(gpufort_host_to_host)),intent(in),value :: direction{{"
  type(c_ptr),value,intent(in) :: stream" if is_async}}
  integer(gpufort_error_kind) :: ierr
  !
  ierr = {{binding}}_(&
    array,c_loc(buffer),memcpy_kind{{",stream" if is_async}})
end function
{%       endfor %}{# async_suffix #}
{%     endfor %}{# datatypes #}
{%   endfor %}{# routines #}
{% endfor %}{# rank #}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_allocate_buffer_routines(gpufort_type,datatypes,max_rank) -%}
{########################################################################################}
{% for target in ["host","device"] %}
{% set is_host = target == "host" %}
{%   for rank in range(1,max_rank+1) %}
{%     set f_kind  = gpufort_type+rank|string %}
{%     set routine = "allocate_"+target+"_buffer" %}
{%     set binding  = f_kind+"_"+routine %}
{%     set size_dims = ",dimension("+rank|string+")" %}
{%     for tuple in datatypes %}
function {{binding}}_{{tuple.f_kind}}(&
    array,buffer{{",&
    pinned,flags" if is_host}}) &
      result(ierr)
  use iso_c_binding
  use gpufort_array_globals
  implicit none
  type({{f_kind}}),intent(in) :: array
  {{tuple.f_type}},intent(inout),pointer,dimension(:{% for i in range(1,rank) %},:{% endfor %}) :: buffer 
{%     if is_host %}
  logical(c_bool),value,optional,intent(in) :: pinned
  integer(c_int),value,optional,intent(in) :: flags
{%     endif %}
  integer(gpufort_error_kind) :: ierr
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
{%- macro render_gpufort_array_deallocate_buffer_routines(gpufort_type,datatypes,max_rank) -%}
{########################################################################################}
{% for target in ["host","device"] %}
{% set is_host = target == "host" %}
{%   for rank in range(1,max_rank+1) %}
{%     set f_kind  = gpufort_type+rank|string %}
{%     set binding  = f_kind+"_"+"deallocate_"+target+"_buffer" %}
{%     set size_dims = ",dimension("+rank|string+")" %}
{%     for tuple in datatypes %}
function {{binding}}_{{tuple.f_kind}}(&
    array,buffer{{",&
    pinned" if is_host}}) &
      result(ierr)
  use iso_c_binding
  use gpufort_array_globals
  implicit none
  type({{f_kind}}),intent(in) :: array
  {{tuple.f_type}},intent(inout),pointer,dimension(:{% for i in range(1,rank) %},:{% endfor %}) :: buffer 
{%     if is_host %}
  logical(c_bool),value,optional,intent(in) :: pinned
{%     endif %}
  integer(gpufort_error_kind) :: ierr
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
  ierr = {{binding}}_(&
    array,c_loc(buffer){{",pinned" if is_host}})
end function
{%     endfor %}{# datatypes #}
{%   endfor %}{# rank #}
{% endfor %}{# target #}
{%- endmacro -%}
{########################################################################################}
