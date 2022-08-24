{# SPDX-License-Identifier: MIT #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{#######################################################################################}
{% macro render_interface(routine,datatypes,max_rank,include_b_variant=True) %}
{#######################################################################################}
interface {{routine}}
{% if include_b_variant %}
  module procedure :: {{routine}}_b
{% endif %}
{% for triple in datatypes %}
  module procedure :: {{routine}}0_{{triple[0]}}
{%   for rank in range(1,max_rank+1) %}
  module procedure :: {{routine}}{{rank}}_{{triple[0]}}
{%   endfor %}{# rank #}
{% endfor %}{# datatypes #}
end interface
{% endmacro %}
{########################################################################################}
{%- macro render_map_routines(datatypes,max_rank) -%}
{# NOTE: type(*) is a Fortran 2018 feature.
{########################################################################################}
{% for clause in ["present","no_create","create","copyin","copy","copyout","delete"] -%}
{%   set routine = "gpufortrt_map_" + clause %}
function {{routine}}_b(hostptr,num_bytes,never_deallocate) result(retval)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  type(c_ptr),intent(in) :: hostptr
  integer(c_size_t),intent(in),optional :: num_bytes 
  logical,intent(in),optional :: never_deallocate
  !
  type(gpufortrt_mapping_t) :: retval
  !
  call gpufortrt_mapping_init(retval,hostptr,num_bytes,&
         gpufortrt_map_kind_{{clause}},never_deallocate)
end function

{%   for triple in datatypes -%}
!> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
function {{routine}}0_{{triple[0]}}(hostptr,never_deallocate) result(retval)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  {{triple[2]}},target,intent(in) :: hostptr
  logical,intent(in),optional :: never_deallocate
  !
  type(gpufortrt_mapping_t) :: retval
  !
  retval = {{routine}}_b(c_loc(hostptr),int({{triple[1]}},c_size_t),never_deallocate)
end function

{%     for rank in range(1,max_rank+1) %}
!> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
function {{routine}}{{rank}}_{{triple[0]}}(hostptr,never_deallocate) result(retval)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  {{triple[2]}},target,intent(in) :: hostptr(:{% for i in range(1,rank) %},:{% endfor %})
  logical,intent(in),optional :: never_deallocate
  !
  type(gpufortrt_mapping_t) :: retval
  !
  retval = {{routine}}_b(c_loc(hostptr),int({{triple[1]}},c_size_t)*size(hostptr),never_deallocate)
end function

{%     endfor %} {# rank #}
{%   endfor %} {# datatypes #}
{% endfor %} {# clause #}
{%- endmacro -%}
{#######################################################################################}
{% macro render_basic_delete_copyout_routines() %}
{#######################################################################################}
{% for clause in ["delete","copyout"] %}
subroutine gpufortrt_{{clause}}_b(hostptr,num_bytes,async_arg,finalize)
  use iso_c_binding
  use gpufortrt_types, only: gpufortrt_handle_kind
  implicit none
  type(c_ptr), intent(in) :: hostptr
  integer(c_size_t),intent(in),optional :: num_bytes
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  logical,intent(in),optional :: finalize
  !
  interface
    subroutine gpufortrt_{{clause}}_c_impl(hostptr,num_bytes) &
        bind(c,name="gpufortrt_{{clause}}")
      use iso_c_binding
      implicit none
      type(c_ptr),value,intent(in) :: hostptr
      integer(c_size_t),value,intent(in) :: num_bytes
    end subroutine
    subroutine gpufortrt_{{clause}}_finalize_c_impl(hostptr,num_bytes) &
        bind(c,name="gpufortrt_{{clause}}_finalize")
      use iso_c_binding
      implicit none
      type(c_ptr),value,intent(in) :: hostptr
      integer(c_size_t),value,intent(in) :: num_bytes
    end subroutine
    subroutine gpufortrt_{{clause}}_async_c_impl(hostptr,num_bytes,async_arg) &
        bind(c,name="gpufortrt_{{clause}}_async")
      use iso_c_binding
      use gpufortrt_types, only: gpufortrt_handle_kind
      implicit none
      type(c_ptr),value,intent(in) :: hostptr
      integer(c_size_t),value,intent(in) :: num_bytes
      integer(gpufortrt_handle_kind),value,intent(in) :: async_arg
    end subroutine
    subroutine gpufortrt_{{clause}}_finalize_async_c_impl(hostptr,num_bytes,async_arg) &
        bind(c,name="gpufortrt_{{clause}}_finalize_async")
      use iso_c_binding
      use gpufortrt_types, only: gpufortrt_handle_kind
      implicit none
      type(c_ptr),value,intent(in) :: hostptr
      integer(c_size_t),value,intent(in) :: num_bytes
      integer(gpufortrt_handle_kind),value,intent(in) :: async_arg
    end subroutine
  end interface
  !
  if ( present(async_arg) ) then
    if ( present(finalize) ) then
      call gpufortrt_{{clause}}_finalize_async_c_impl(hostptr,num_bytes,async_arg)
    else
      call gpufortrt_{{clause}}_async_c_impl(hostptr,num_bytes,async_arg)
    endif
  else
    if ( present(finalize) ) then
      call gpufortrt_{{clause}}_finalize_c_impl(hostptr,num_bytes)
    else
      call gpufortrt_{{clause}}_c_impl(hostptr,num_bytes)
    endif
  endif
end subroutine

{% endfor %}
{% endmacro %}
{#######################################################################################}
{% macro render_specialized_delete_copyout_routines(datatypes,max_rank) %}
{#######################################################################################}
{% for clause in ["delete","copyout"] %}
{%   for triple in datatypes -%}
!> (Specialized version for Fortran scalar arguments)
subroutine gpufortrt_{{clause}}0_{{triple[0]}}(hostptr,async_arg,finalize)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  {{triple[2]}},target,intent(in) :: hostptr
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  logical,intent(in),optional :: finalize
  !
  call gpufortrt_{{clause}}_b(c_loc(hostptr),int({{triple[1]}},kind=c_size_t),&
                              async_arg,finalize)
end subroutine

{%     for rank in range(1,max_rank+1) %}
!> (Specialized version for Fortran array arguments)
subroutine gpufortrt_{{clause}}{{rank}}_{{triple[0]}}(hostptr,async_arg,finalize)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  {{triple[2]}},target,intent(in) :: hostptr(:{% for i in range(1,rank) %},:{% endfor %})
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  logical,intent(in),optional :: finalize
  !
  call gpufortrt_{{clause}}_b(c_loc(hostptr),int({{triple[1]}},kind=c_size_t)*size(hostptr),&
                              async_arg,finalize)
end subroutine

{%     endfor %}{# rank #}
{%   endfor %}{# datatypes #}
{% endfor %}{# clause #}
{% endmacro %}
{#######################################################################################}
{% macro render_basic_copy_routines() %}
{#######################################################################################}
{% for clause in ["create","copyin","copy"] %}
!> Map and directly return the corresponding deviceptr.
function gpufortrt_{{clause}}_b(hostptr,num_bytes,never_deallocate,&
    async_arg) result(deviceptr)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  type(c_ptr),intent(in) :: hostptr
  integer(c_size_t),intent(in) :: num_bytes
  logical,intent(in),optional :: never_deallocate
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
  !
  type(c_ptr) :: deviceptr
  !
  interface
    function gpufortrt_{{clause}}_c_impl(hostptr,num_bytes,never_deallocate) &
        bind(c,name="gpufortrt_{{clause}}") result(deviceptr)
      use iso_c_binding
      use gpufortrt_types
      implicit none
      type(c_ptr),value,intent(in) :: hostptr
      integer(c_size_t),value,intent(in) :: num_bytes
      logical(c_bool),value,intent(in) :: never_deallocate
      !
      type(c_ptr) :: deviceptr
    end function
    function gpufortrt_{{clause}}_async_c_impl(hostptr,num_bytes,never_deallocate,async_arg) &
        bind(c,name="gpufortrt_{{clause}}_async") result(deviceptr)
      use iso_c_binding
      use gpufortrt_types
      implicit none
      type(c_ptr),value,intent(in) :: hostptr
      integer(c_size_t),value,intent(in) :: num_bytes
      logical(c_bool),value,intent(in) :: never_deallocate
      integer(gpufortrt_handle_kind),value,intent(in) :: async_arg
      !
      type(c_ptr) :: deviceptr
    end function
  end interface
  !
  logical(c_bool) :: opt_never_deallocate
  !
  opt_never_deallocate = .false._c_bool
  if ( present(never_deallocate) ) opt_never_deallocate = never_deallocate
  if ( present(async_arg) ) then
    deviceptr = gpufortrt_{{clause}}_async_c_impl(hostptr,num_bytes,opt_never_deallocate,async_arg)
  else
    deviceptr = gpufortrt_{{clause}}_c_impl(hostptr,num_bytes,opt_never_deallocate)
  endif
end function

{% endfor %}
{% endmacro %}
{#######################################################################################}
{% macro render_specialized_copy_routines(datatypes,max_rank) %}
{#######################################################################################}
{% for clause in ["create","copyin","copy"] %}
{%   for triple in datatypes -%}
!> Map and directly return the corresponding deviceptr.
!> (Specialized version for Fortran scalar arguments)
function gpufortrt_{{clause}}0_{{triple[0]}}(hostptr,never_deallocate,async_arg) result(deviceptr)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  {{triple[2]}},target,intent(in) :: hostptr
  logical,intent(in),optional :: never_deallocate
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
  !
  type(c_ptr) :: deviceptr
  !
  deviceptr = gpufortrt_{{clause}}_b(c_loc(hostptr),int({{triple[1]}},kind=c_size_t),&
                                     never_deallocate,async_arg)
end function

{%     for rank in range(1,max_rank+1) %}
!> Map and directly return the corresponding deviceptr.
!> (Specialized version for Fortran array arguments)
function gpufortrt_{{clause}}{{rank}}_{{triple[0]}}(hostptr,&
    never_deallocate,async_arg) result(deviceptr)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  {{triple[2]}},target,intent(in) :: hostptr(:{% for i in range(1,rank) %},:{% endfor %})
  logical,intent(in),optional :: never_deallocate
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
  !
  type(c_ptr) :: deviceptr
  !
  deviceptr = gpufortrt_{{clause}}_b(c_loc(hostptr),int({{triple[1]}},kind=c_size_t)*size(hostptr),&
                                     never_deallocate,async_arg)
end function
{%     endfor %}{# rank #}
{%   endfor %}{# datatypes #}
{% endfor %}{# clause #}
{% endmacro %}
{#######################################################################################}
{% macro render_specialized_deviceptr_routines(datatypes,max_rank) %}
{#######################################################################################}
{% for triple in datatypes -%}
function gpufortrt_deviceptr0_{{triple[0]}}(hostptr) result(deviceptr)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  {{triple[2]}},target,intent(in) :: hostptr
  !
  type(c_ptr) :: deviceptr
  !
  deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
end function

{%   for rank in range(1,max_rank+1) %}
function gpufortrt_deviceptr{{rank}}_{{triple[0]}}(hostptr) result(deviceptr)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  {{triple[2]}},target,intent(in) :: hostptr(:{% for i in range(1,rank) %},:{% endfor %})
  !
  type(c_ptr) :: deviceptr
  !
  deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
end function

{%   endfor %}{# rank #}
{% endfor %}{# datatypes #}
{% endmacro %}
{#######################################################################################}
{% macro render_specialized_present_routines(datatypes,max_rank) %}
{#######################################################################################}
{% for clause in ["present"] %}
{%   for triple in datatypes -%}
function gpufortrt_{{clause}}0_{{triple[0]}}(hostptr) result(deviceptr)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  {{triple[2]}},target,intent(in) :: hostptr
  !
  type(c_ptr) :: deviceptr
  !
  deviceptr = gpufortrt_{{clause}}_b(c_loc(hostptr),int({{triple[1]}},kind=c_size_t))
end function

{%     for rank in range(1,max_rank+1) %}
function gpufortrt_{{clause}}{{rank}}_{{triple[0]}}(hostptr) result(deviceptr)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  {{triple[2]}},target,intent(in) :: hostptr(:{% for i in range(1,rank) %},:{% endfor %})
  !
  type(c_ptr) :: deviceptr
  !
  deviceptr = gpufortrt_{{clause}}_b(c_loc(hostptr),int({{triple[1]}},kind=c_size_t)*size(hostptr))
end function

{%     endfor %}{# rank #}
{%   endfor %}{# datatypes #}
{% endfor %}{# clause #}
{% endmacro %}
{#######################################################################################}
{% macro render_map_interfaces(datatypes,max_rank) %}
{#######################################################################################}
{% for clause in ["present","no_create","create","copyin","copy","copyout","delete"] %}
{{ render_interface("gpufortrt_map_"+clause,datatypes,max_rank) }}
{% endfor %}
{% endmacro %}
{#######################################################################################}
{% macro render_copy_interfaces(datatypes,max_rank) %}
{#######################################################################################}
{% for clause in ["present"] %}
{{ render_interface("gpufortrt_"+clause,datatypes,max_rank,False) }}

{% endfor %}
{% for clause in ["create","copyin","copy"] %}
{{ render_interface("gpufortrt_"+clause,datatypes,max_rank) }}

{% endfor %}
{% endmacro %}
{#######################################################################################}
{% macro render_basic_update_routine(update_kind) %}
{#######################################################################################}
subroutine gpufortrt_update_{{update_kind}}_b(hostptr,num_bytes,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  type(c_ptr),intent(in) :: hostptr
  integer(c_size_t),intent(in) :: num_bytes
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  interface
    subroutine gpufortrt_update_{{update_kind}}_c_impl(hostptr,num_bytes,if_arg,if_present_arg) &
            bind(c,name="gpufortrt_update_{{update_kind}}")
      use iso_c_binding
      implicit none
      type(c_ptr),value,intent(in) :: hostptr
      integer(c_size_t),value,intent(in) :: num_bytes
      logical(c_bool),value,intent(in) :: if_arg, if_present_arg
    end subroutine
    subroutine gpufortrt_update_{{update_kind}}_async_c_impl(hostptr,num_bytes,if_arg,if_present_arg,async_arg) &
            bind(c,name="gpufortrt_update_{{update_kind}}_async")
      use iso_c_binding
      use gpufortrt_types
      implicit none
      type(c_ptr),value,intent(in) :: hostptr
      integer(c_size_t),value,intent(in) :: num_bytes
      logical(c_bool),value,intent(in) :: if_arg, if_present_arg
      integer(gpufortrt_handle_kind),value,intent(in) :: async_arg
    end subroutine
  end interface
  logical :: opt_if_arg, opt_if_present_arg
  !
  opt_if_arg = .true.
  opt_if_present_arg = .false.
  if ( present(if_arg) ) opt_if_arg = if_arg
  if ( present(if_present_arg) ) opt_if_present_arg = if_present_arg
  !
  if ( present(async_arg) ) then
    call gpufortrt_update_{{update_kind}}_async_c_impl(hostptr,&
                                                       num_bytes,&
                                                       logical(opt_if_arg,c_bool),&
                                                       logical(opt_if_present_arg,c_bool),&
                                                       async_arg)
  else
    call gpufortrt_update_{{update_kind}}_c_impl(hostptr,&
                                                 num_bytes,&
                                                 logical(opt_if_arg,c_bool),&
                                                 logical(opt_if_present_arg,c_bool))
  endif
end subroutine
{% endmacro %}
{#######################################################################################}
{% macro render_specialized_update_routines(update_kind,datatypes,max_rank) %}
{#######################################################################################}
{% for triple in datatypes %}
subroutine gpufortrt_update_{{update_kind}}0_{{triple[0]}}(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  {{triple[2]}},target,intent(in) :: hostptr
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_{{update_kind}}_b(c_loc(hostptr),int({{triple[1]}},c_size_t),if_arg,if_present_arg,async_arg)
end subroutine

{%   for rank in range(1,max_rank+1) %}
subroutine gpufortrt_update_{{update_kind}}{{rank}}_{{triple[0]}}(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  {{triple[2]}},target,intent(in) :: hostptr(:{% for i in range(1,rank) %},:{% endfor %})
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_{{update_kind}}_b(c_loc(hostptr),int({{triple[1]}},c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

{%   endfor %}{# rank #}
{% endfor %}{# datatypes #}
{% endmacro %}
{########################################################################################}
{%- macro render_set_fptr_lower_bound(fptr,rank) -%}
{########################################################################################}
{{fptr}}(&
{% for i in range(1,rank+1) %}
  opt_lbounds({{i}}):{{ "," if not loop.last else ")" }}&
{% endfor %}
    => {{fptr}}
{%- endmacro  -%}
{#######################################################################################}
{% macro render_specialized_use_device_routines(datatypes,max_rank) %}
{#######################################################################################}
{% for triple in datatypes %}
subroutine gpufortrt_use_device0_{{triple[0]}}(resultptr,hostptr,if_arg,if_present_arg)
  use iso_c_binding
  implicit none
  {{triple[2]}},target,intent(in) :: hostptr
  logical,intent(in),optional :: if_arg, if_present_arg
  !
  {{triple[2]}},pointer,intent(inout) :: resultptr
  !
  type(c_ptr) :: tmp_cptr
  !
  tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
  call c_f_pointer(tmp_cptr,resultptr)
end subroutine

{% for rank in range(1,max_rank+1) %}
subroutine gpufortrt_use_device{{rank}}_{{triple[0]}}(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
  use iso_c_binding
  implicit none
  {{triple[2]}},target,intent(in) :: hostptr({% for i in range(1,rank) %}1:1,{% endfor %}*)
  integer,intent(in),optional :: sizes({{rank}}), lbounds({{rank}})
  logical,intent(in),optional :: if_arg, if_present_arg
  !
  {{triple[2]}},pointer,intent(inout) :: resultptr(:{% for i in range(1,rank) %},:{% endfor %})
  !
  type(c_ptr) :: tmp_cptr
  integer :: opt_sizes({{rank}}), opt_lbounds({{rank}})
  !
  opt_sizes = 1
  opt_lbounds = 1  
  if ( present(sizes) ) opt_sizes = sizes
  if ( present(lbounds) ) opt_lbounds = lbounds
  tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
  call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
{{ render_set_fptr_lower_bound("resultptr",rank) | indent(2,True) }}
end subroutine

{%   endfor %}{# rank #}
{% endfor %}{# datatypes #}
{% endmacro %}
{#######################################################################################}
{% macro render_use_device_interface(datatypes,max_rank) %}
{#######################################################################################}
{% for rank in range(0,max_rank+1) %}
interface gpufortrt_use_device{{rank}}
{%   for triple in datatypes %}
  module procedure :: gpufortrt_use_device{{rank}}_{{triple[0]}}
{%   endfor %}{# datatypes #}
end interface
{% endfor %}{# rank #}
{% endmacro %}
{#######################################################################################}