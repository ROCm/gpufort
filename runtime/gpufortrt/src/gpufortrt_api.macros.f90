{# SPDX-License-Identifier: MIT #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{########################################################################################}
{%- macro render_set_fptr_lower_bound(fptr,array,rank) -%}
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
{#######################################################################################}
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
{%- macro render_map_dec_struct_refs_routines(datatypes) -%}
{# NOTE: type(*) is a Fortran 2018 feature.
{########################################################################################}
{% for tuple in datatypes -%}
function gpufortrt_map_dec_struct_refs_{{tuple[0]}}_scal(hostptr) result(retval)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  {{tuple[2]}},target,intent(in) :: hostptr
  !
  type(mapping_t) :: retval
  !
  call gpufort_mapping_init(retval,c_loc(hostptr),0_c_size_t,gpufortrt_map_kind_dec_struct_refs,.false.)
end function

function gpufortrt_map_dec_struct_refs_{{tuple[0]}}_arr(hostptr) result(retval)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  {{tuple[2]}},dimension(*),target,intent(in) :: hostptr
  !
  type(mapping_t) :: retval
  !
  call gpufort_mapping_init(retval,c_loc(hostptr),0_c_size_t,gpufortrt_map_kind_dec_struct_refs,.false.)
end function
{% endfor %}
{########################################################################################}
{%- macro render_map_routines(data_clauses,datatypes) -%}
{# NOTE: type(*) is a Fortran 2018 feature.
{########################################################################################}
{% for clause in data_clauses -%}
{%   set is_never_deallocate_clause = clause in ["create","copyin","copy","copyout"] %} {# plus: device_resident, link #}
{%   set extra_args = ",never_deallocate" if is_never_deallocate_clause else "" %}
{%   set routine = "gpufortrt_map_" + clause %}
{% for tuple in datatypes -%}
function {{routine}}_{{tuple[0]}}_scal(hostptr{{extra_args}}) result(retval)
  use iso_c_binding
  use gpufortrt_types
  ues gpufortrt_auxiliary
  implicit none
  {{tuple[2]}},target,intent(in) :: hostptr
{%     if is_never_deallocate_clause %}
   logical,intent(in),optional :: never_deallocate
{%     endif %}
  !
  type(mapping_t) :: retval
  !
  call gpufortrt_mapping_init(retval,c_loc(hostptr),int({{tuple[1]}},c_size_t){{extra_args}})
end function

function {{routine}}_{{tuple[0]}}_arr(hostptr,num_elements,{{extra_args}}) result(retval)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  {{tuple[2]}},dimension(*),target,intent(in) :: hostptr
  integer,intent(in) :: num_elements
{%     if is_never_deallocate_clause %}
   logical,intent(in),optional :: never_deallocate
{%     endif %}
  !
  type(mapping_t) :: retval
  !
  call gpufortrt_mapping_init(retval,retval,c_loc(hostptr),&
    int({{tuple[1]}},c_size_t)*int(num_elements,c_size_t){{extra_args}})
end function
{% endfor -%} 
{%- endmacro -%}
{#######################################################################################}
{% macro render_basic_map_and_lookup_routines() %}
{#######################################################################################}
{% for clause in ["create","copyin","copy","copyout"] %}
!> Map and directly return the corresponding deviceptr.
function gpufortrt_{{clause}}_b(hostptr,never_deallocate,&
    async_arg,ctr_to_update) result(deviceptr)
  use iso_c_binding
  use gpufortrt_auxiliary
  implicit none
  type(c_ptr),intent(in) :: hostptr
  integer(c_size_t),intent(in) :: num_bytes
  logical,intent(in),optional :: never_deallocate
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
  type(gpufortrt_counter_t),optional,intent(in) :: ctr_to_update
  !
  type(c_ptr) :: deviceptr
  !
  interface
    subroutine gpufortrt_{{clause}}_c_impl(hostptr,num_bytes,never_deallocate,ctr_to_update) &
        bind(c,name="gpufortrt_{{clause}}") result(deviceptr)
      type(c_ptr),value,intent(in) :: hostptr
      integer(c_size_t),value,intent(in) :: num_bytes
      logical(c_bool),intent(in),optional :: never_deallocate
      type(gpufortrt_counter_t),value,intent(in) :: ctr_to_update
    end subroutine
    subroutine gpufortrt_{{clause}}_async_c_impl(hostptr,num_bytes,never_deallocate,async_arg,ctr_to_update) &
        bind(c,name="gpufortrt_{{clause}}_async") result(deviceptr)
      type(c_ptr),value,intent(in) :: hostptr
      integer(c_size_t),value,intent(in) :: num_bytes
      logical(c_bool),intent(in),optional :: never_deallocate
      integer(gpufortrt_handle_kind),value,intent(in) :: async_arg
      type(gpufortrt_counter_t),value,intent(in) :: ctr_to_update
    end subroutine
  end interface
  !
  if ( present(async_arg) ) then
    deviceptr = gpufortrt_{{clause}}_async_c_impl(hostptr,num_bytes,async_arg,&
        eval_optval(ctr_to_update,gpufortrt_counter_none))
  else
    deviceptr = gpufortrt_{{clause}}_c_impl(hostptr,num_bytes,&
        eval_optval(ctr_to_update,gpufortrt_counter_none))
  endif
end function
{% endfor %}
{% endmacro %}
{#######################################################################################}
{% macro render_specialized_map_and_lookup_routines(datatypes) %}
{#######################################################################################}
{% for clause in ["create","copyin","copy","copyout"] %}
{%   for tuple in datatypes -%}
!> Map and directly return the corresponding deviceptr.
!> (Specialized version for Fortran scalar arguments)
function gpufortrt_{{clause}}_{{tuple[0]}}_scal(hostptr,never_deallocate,async_arg,ctr_to_update) result(deviceptr)
  use iso_c_binding
  implicit none
  {{tuple[2]}},target,intent(in) :: hostptr
  logical,intent(in),optional :: never_deallocate
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
  type(gpufortrt_counter_t),optional,intent(in) :: ctr_to_update
  !
  type(c_ptr) :: deviceptr
  !
  deviceptr = gpufortrt_{{clause}}_b(c_loc(hostptr),int({{tuple[1]}},kind=c_size_t),&
                                     never_deallocate,async_arg,ctr_to_update)
end function

!> Map and directly return the corresponding deviceptr.
!> (Specialized version for Fortran array arguments)
function gpufortrt_{{clause}}_{{tuple[0]}}_arr(hostptr,num_elements,&
    never_deallocate,async_arg,ctr_to_update) result(deviceptr)
  use iso_c_binding
  implicit none
  {{tuple[2]}},dimension(*),target,intent(in) :: hostptr
  integer,intent(in) :: num_elements
  logical,intent(in),optional :: never_deallocate
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
  type(gpufortrt_counter_t),intent(in),optional :: ctr_to_update
  !
  type(c_ptr) :: deviceptr
  !
  deviceptr = gpufortrt_{{clause}}_b(c_loc(hostptr),int(num_elements,kind=c_size_t)*int({{tuple[1]}},kind=c_size_t),&
                                     never_deallocate,async_arg,ctr_to_update)
end function
{%   endfor %}{# datatypes #}
{% endfor %}{# clause #}
{% endmacro %}
{#######################################################################################}
{% macro render_specialized_present_routines(datatypes) %}
{#######################################################################################}
{% for tuple in datatypes -%}
function gpufortrt_present_{{tuple[0]}}_scal(hostptr,ctr_to_update) result(deviceptr)
  use iso_c_binding
  use gpufortrt_auxiliary
  implicit none
  {{tuple[2]}},target,intent(in) :: hostptr
  type(gpufortrt_counter_t),optional,intent(in) :: ctr_to_update
  !
  type(c_ptr) :: deviceptr
  !
  deviceptr = gpufortrt_present_b(c_loc(hostptr),int({{tuple[1]}},kind=c_size_t),&
                                  eval_optval(ctr_to_update,gpufortrt_counter_none))
end function

function gpufortrt_present_{{tuple[0]}}_arr(hostptr,ctr_to_update) result(deviceptr)
  use iso_c_binding
  use gpufortrt_auxiliary
  implicit none
  {{tuple[2]}},target,intent(in) :: hostptr
  integer,intent(in) :: num_elements
  type(gpufortrt_counter_t),optional,intent(in) :: ctr_to_update
  !
  type(c_ptr) :: deviceptr
  !
  deviceptr = gpufortrt_present_b(c_loc(hostptr),int({{tuple[1]}},kind=c_size_t),&
                                  eval_optval(ctr_to_update,gpufortrt_counter_none))
end function
{% endfor %}{# datatypes #}
{#######################################################################################}
{% macro render_map_and_lookup_interfaces(datatypes) %}
{#######################################################################################}
{% for clause in ["create","copyin","copy","copyout"] %}
interface
  module procedure ::  
    gpufortrt_{{clause}}_{{tuple[0]}}_b,&
{%   for tuple in datatypes -%}
    gpufortrt_{{clause}}_{{tuple[0]}}_scal,&
    gpufortrt_{{clause}}_{{tuple[0]}}_arr{{ ",&" if not loop.last }}{%   
     endfor %}{{ ",&" if not loop.last }}{# datatypes #}
{% endfor %}{# clause #}
end interface
{% endmacro %}
{#######################################################################################}
{% macro render_use_device_routines() %}

{% endmacro %}
{#######################################################################################}
