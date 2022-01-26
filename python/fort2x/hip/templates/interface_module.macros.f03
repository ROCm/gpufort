{#- SPDX-License-Identifier: MIT                                        -#}
{#- Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.-#}
{########################################################################################}
{% import "common.macros.f03" as cm %}
{########################################################################################}
{%- macro render_launcher(kernel,
                          launcher) -%}
{%- set num_global_vars = kernel.global_vars|length + kernel.global_reduced_vars|length -%}
function {{launcher.name}}(&
{% if launcher.kind in ["hip"] %}
    grid, block,&
{% endif %}
    sharedmem,stream{{"," if num_global_vars > 0 }}&
{{cm.render_params(kernel.global_vars+kernel.global_reduced_vars,sep=",&\n") | indent(4,True)}}) bind(c,name="{{launcher.name}}_") &
      result(ierr)
  use iso_c_binding
  use hipfort
{% if launcher.used_modules|length %}
{{ render_used_modules(launcher.used_modules) | indent(2,True) }}
{% endif %}
  implicit none
{% if launcher.kind in ["hip"] %}  
  type(dim3),intent(in) :: grid, block
{% endif %}
  integer(c_int),intent(in) :: sharedmem
  type(c_ptr),intent(in)    :: stream{{"," if num_global_vars > 0}}
{{ cm.render_param_decls(kernel.global_vars+kernel.global_reduced_vars) | indent(2,True) }}
  integer(kind(hipSuccess)) :: ierr
end function
{%- endmacro -%}
{########################################################################################}
{%- macro render_cpu_routine(kernel,
                             launcher,
                             fortran_snippet) -%}
{%- set num_global_vars = kernel.global_vars|length + kernel.global_reduced_vars|length -%}
function {{kernel.name}}_cpu(&
    sharedmem,stream{{"," if num_global_vars > 0 }}&
{{cm.render_params(kernel.global_vars+kernel.global_reduced_vars,sep=",&\n") | indent(4,True)}}) bind(c,name="{{kernel.name}}_cpu") &
      result(ierr)
  use iso_c_binding
  use hipfort
{% if launcher.used_modules|length %}
{{ render_used_modules(launcher.used_modules) | indent(2,True) }}
{% endif %}
  implicit none
{% if launcher.kind in ["hip"] %}
  type(dim3),intent(in) :: grid, block
{% endif %}
  integer(c_int),intent(in) :: sharedmem
  type(c_ptr),intent(in)    :: stream{{"," if num_global_vars > 0}}
{{ cm.render_param_decls(kernel.global_vars+kernel.global_reduced_vars,"","_in") | indent(2,True) }}
  integer(kind(hipSuccess)) :: ierr
{# TODO 
 - rename gpufort_array inputs:
   - give suffix _in
 - create local arrays
 - copy device data to local arrays from input gpufort arrays
 - run code
 - copy local host array data back to device data
 - destroy local arrays
#}
  !
{{ fortran_snippet | indent(2,True) }}
end function
{%- endmacro -%}