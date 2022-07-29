{# SPDX-License-Identifier: MIT #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{########################################################################################}
{%- set data_clauses = [
    "delete",
    "present",
    "no_create",
    "create",
    "copy",
    "copyin",
    "copyout",
  ]
-%}
{########################################################################################}
{%- macro render_map_routines(data_clauses,datatypes,dimensions) -%}
{# NOTE: type(*) is a Fortrana 2018 feature.
{########################################################################################}
{% for tuple in datatypes -%}
{%   set suffix = tuple[0] %}                                                              
function gpufortrt_map_dec_struct_refs_{{suffix}}_scal(hostptr) result(retval)
  use iso_c_binding
  use gpufortrt_base, only: mapping_t
  implicit none
  {{tuple[2]}},target,intent(in) :: hostptr
  !
  type(mapping_t) :: retval
  !
  call retval%init(c_loc(hostptr),0_c_size_T,gpufortrt_map_kind_dec_struct_refs,.false.)
end function

function gpufortrt_map_dec_struct_refs_{{suffix}}_arr(hostptr) result(retval)
  use iso_c_binding
  use gpufortrt_base, only: mapping_t
  implicit none
  {{tuple[2]}},dimension(*),target,intent(in) :: hostptr
  !
  type(mapping_t) :: retval
  !
  call retval%init(c_loc(hostptr),0_c_size_t,gpufortrt_map_kind_dec_struct_refs,.false.)
end function
{% endfor %}

{% for clause in data_clauses -%}
{% set is_acc_declare_module_clause = clause in ["create","copyin"] %} {# plus: device_resident, link #}
{% set extra_args = ",declared_module_var" if is_acc_declare_module_clause else "" %}
{% set routine = "gpufortrt_map_" + clause %}
! {{routine}}
function {{routine}}_b(hostptr,num_bytes{{extra_args}}) result(retval)
  use iso_c_binding
  use gpufortrt_base, only: mapping_t
  implicit none
  !
  type(c_ptr),intent(in)       :: hostptr
  integer(c_size_t),intent(in) :: num_bytes
{% if is_acc_declare_module_clause %}
   logical,intent(in),optional :: declared_module_var
{% endif %}
  !
  type(mapping_t) :: retval
  !
  logical :: opt_declared_module_var
  !
  opt_declared_module_var = .false.
{% if is_acc_declare_module_clause %}
  if ( present(declared_module_var) ) opt_declared_module_var = declared_module_var
{% endif %}
  call retval%init(hostptr,num_bytes,gpufortrt_map_kind_{{clause}},opt_declared_module_var)
end function

{% for tuple in datatypes -%}
{%-  for dims in dimensions -%}
{%     if dims > 0 %}
{%       set size = 'size(hostptr)*' %}
{%       set rank = ',dimension(' + ':,'*(dims-1) + ':)' %}
{%     else %}
{%       set size = '' %}
{%       set rank = '' %}
{%     endif %}
{%     set suffix = tuple[0] + "_" + dims|string %}                                                              
function {{routine}}_{{suffix}}(hostptr{{extra_args}}) result(retval)
  use iso_c_binding
  use gpufortrt_base, only: mapping_t
  implicit none
  {{tuple[2]}},target{{ rank }},intent(in) :: hostptr
{%     if is_acc_declare_module_clause %}
   logical,intent(in),optional :: declared_module_var
{%     endif %}
  !
  type(mapping_t) :: retval
  !
  retval = {{routine}}_b(c_loc(hostptr),int({{size}}{{tuple[1]}},c_size_t){{extra_args}})
end function

{% endfor %} 
{% endfor %} 
{% endfor -%} 
{%- endmacro -%}
{########################################################################################}
{%- macro render_set_fptr_lower_bound(fptr,
                                     array,
                                     rank) -%}
{########################################################################################}
{% set rank_ub=rank+1 %}
{{fptr}}(&
{% for i in range(1,rank_ub) %}
  lbound({{array}},{{i}}):{{ "," if not loop.last else ")" }}&
{% endfor %}
    => {{fptr}}
{%- endmacro  -%}
{#######################################################################################}
{% macro render_interface(interface_name,has_backend=False) %}
interface {{interface_name}}
  module procedure &
{% if has_backend %}
    {{interface_name}}_b, &
{% endif %}
{% for tuple in datatypes %}
{%   for dims in dimensions %}
{%     set name = interface_name+"_" + tuple[0] + "_" + dims|string %}
    {{name}}{{",&\n" if not loop.last}}{% endfor %}{{ ",&" if not loop.last}}
{% endfor %}
end interface
{% endmacro %}
{########################################################################################}
! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.

! TODO not thread-safe, make use of OMP_LIB module if this becomes a problem

! TODO will be slow for large pointer storages
! If this becomes an issue, use faster data structures or bind the public interfaces to
! a faster C runtime


module gpufortrt
  use gpufortrt_base
    
  !> Lookup device pointer for given host pointer.
  !> \param[in] condition condition that must be met, otherwise host pointer is returned. Defaults to '.true.'.
  !> \param[in] if_present Only return device pointer if one could be found for the host pointer.
  !>                       otherwise host pointer is returned. Defaults to '.false.'.
  !> \note Returns a c_null_ptr if the host pointer is invalid, i.e. not C associated.
{{ render_interface("gpufortrt_use_device",True) | indent(2,True) }}

  !> Decrement the structured reference counter of a record.
  interface gpufortrt_map_dec_struct_refs
{% for tuple in datatypes -%}
    module procedure :: gpufortrt_map_dec_struct_refs_{{tuple[0]}}_scal,&
                        gpufortrt_map_dec_struct_refs_{{tuple[0]}}_arr
{% endfor %}
  end interface

{% for mapping in data_clauses %} 
{{ render_interface("gpufortrt_"+mapping,True) | indent(2,True) }}

{% endfor %}
{% for mapping in data_clauses %} 
{{ render_interface("gpufortrt_map_"+mapping,True) | indent(2,True) }}

{% endfor %}

  !> Update Directive
  !>
  !> The update directive copies data between the memory for the
  !> encountering thread and the device. An update directive may
  !> appear in any data region, including an implicit data region.
  !>
  !> FORTRAN
  !>
  !> !$acc update [clause [[,] clause]…]
  !>
  !> CLAUSES
  !>
  !> self( list ) or host( list )
  !>   Copies the data in list from the device to the encountering
  !>   thread.
  !> device( list )
  !>   Copies the data in list from the encountering thread to the
  !>   device.
  !> if( condition )
  !>   When the condition is zero or .FALSE., no data will be moved to
  !>   or from the device.
  !> if_present
  !>   Issue no error when the data is not present on the device.
  !> async [( expression )]
  !>   The data movement will execute asynchronously with the
  !>   encountering thread on the corresponding async queue.
  !> wait [( expression-list )]
  !>   The data movement will not begin execution until all actions on
  !>   the corresponding async queue(s) are complete.
{{ render_interface("gpufortrt_update_host",True) | indent(2,True) }}

  !> Update Directive
  !>
  !> The update directive copies data between the memory for the
  !> encountering thread and the device. An update directive may
  !> appear in any data region, including an implicit data region.
  !>
  !> FORTRAN
  !>
  !> !$acc update [clause [[,] clause]…]
  !>
  !> CLAUSES
  !>
  !> self( list ) or host( list )
  !>   Copies the data in list from the device to the encountering
  !>   thread.
  !> device( list )
  !>   Copies the data in list from the encountering thread to the
  !>   device.
  !> if( condition )
  !>   When the condition is zero or .FALSE., no data will be moved to
  !>   or from the device.
  !> if_present
  !>   Issue no error when the data is not present on the device.
  !> async [( expression )]
  !>   The data movement will execute asynchronously with the
  !>   encountering thread on the corresponding async queue.
  !> wait [( expression-list )]
  !>   The data movement will not begin execution until all actions on
  !>   the corresponding async queue(s) are complete.
{{ render_interface("gpufortrt_update_device",True) | indent(2,True) }}

  contains

    !
    ! autogenerated routines for different inputs
    !

{% for tuple in datatypes -%}
{%-  for dims in dimensions -%}
{%     if dims > 0 %}
{%       set shape = ',shape(hostptr)' %}
{%       set size = 'size(hostptr)*' %}
{%       set rank = ',dimension(' + ':,'*(dims-1) + ':)' %}
{%     else %}
{%       set shape = '' %}
{%       set size = '' %}
{%       set rank = '' %}
{%     endif %}
{%     set suffix = tuple[0] + "_" + dims|string %}
    function gpufortrt_use_device_{{suffix}}(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_base, only: gpufortrt_use_device_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(in) :: hostptr
      integer(c_int),dimension({{ dims }}),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      {{tuple[2]}},pointer{{ rank }} :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),{{size}}{{tuple[1]}}_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr{{shape}})
{%     if dims > 0 %}
      if ( present(lbounds) ) then
{{ render_set_fptr_lower_bound("resultptr","hostptr",dims) | indent(8,True) }}
      endif
{%     endif %}
    end function 

    function gpufortrt_present_{{suffix}}(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_base, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      {{tuple[2]}},target{{ rank }},intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),{{size}}{{tuple[1]}}_c_size_t)
    end function

    function gpufortrt_create_{{suffix}}(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_base, only: gpufortrt_create_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),{{size}}{{tuple[1]}}_c_size_t)
    end function

    function gpufortrt_no_create_{{suffix}}(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_base, only: gpufortrt_no_create_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_{{suffix}}(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_base, only: gpufortrt_delete_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_{{suffix}}(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_base, only: gpufortrt_copyin_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),{{size}}{{tuple[1]}}_c_size_t,async)
    end function

    function gpufortrt_copyout_{{suffix}}(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_base, only: gpufortrt_copyout_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),{{size}}{{tuple[1]}}_c_size_t,async)
    end function

    function gpufortrt_copy_{{suffix}}(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_base, only: gpufortrt_copy_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),{{size}}{{tuple[1]}}_c_size_t,async)
    end function

{% for target in ["host","device"] %}
    subroutine gpufortrt_update_{{target}}_{{suffix}}(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_base, only: gpufortrt_update_host_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_{{target}}_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
{% endfor %}
{% endfor %}
{% endfor %}

{{ render_map_routines(data_clauses,datatypes,dimensions) | indent(2,True) }}
end module