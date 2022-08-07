{# SPDX-License-Identifier: MIT #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{########################################################################################}
{%- macro render_map_routines(data_clauses,datatypes,dimensions) -%}
{# NOTE: type(*) is a Fortran 2018 feature.
{########################################################################################}
{% for tuple in datatypes -%}
{%   set suffix = tuple[0] %}                                                              
function gpufortrt_map_dec_struct_refs_{{suffix}}_scal(hostptr) result(retval)
  use iso_c_binding
  use gpufortrt_core, only: mapping_t
  implicit none
  {{tuple[2]}},target,intent(in) :: hostptr
  !
  type(mapping_t) :: retval
  !
  call retval%init(c_loc(hostptr),0_c_size_T,gpufortrt_map_kind_dec_struct_refs,.false.)
end function

function gpufortrt_map_dec_struct_refs_{{suffix}}_arr(hostptr) result(retval)
  use iso_c_binding
  use gpufortrt_core, only: mapping_t
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
{% set extra_args = ",never_deallocate" if is_acc_declare_module_clause else "" %}
{% set routine = "gpufortrt_map_" + clause %}
! {{routine}}
function {{routine}}_b(hostptr,num_bytes{{extra_args}}) result(retval)
  use iso_c_binding
  use gpufortrt_core, only: mapping_t
  implicit none
  !
  type(c_ptr),intent(in)       :: hostptr
  integer(c_size_t),intent(in) :: num_bytes
{% if is_acc_declare_module_clause %}
   logical,intent(in),optional :: never_deallocate
{% endif %}
  !
  type(mapping_t) :: retval
  !
  logical :: opt_never_deallocate
  !
  opt_never_deallocate = .false.
{% if is_acc_declare_module_clause %}
  if ( present(never_deallocate) ) opt_never_deallocate = never_deallocate
{% endif %}
  call retval%init(hostptr,num_bytes,gpufortrt_map_kind_{{clause}},opt_never_deallocate)
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
  use gpufortrt_core, only: mapping_t
  implicit none
  {{tuple[2]}},target{{ rank }},intent(in) :: hostptr
{%     if is_acc_declare_module_clause %}
   logical,intent(in),optional :: never_deallocate
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
