{# SPDX-License-Identifier: MIT                                                 #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{########################################################################################}
{% import "common.macros.f03" as cm %}
{########################################################################################}
{% macro render_interface(name,datatypes,max_rank=0) %}
interface {{name}}
  procedure :: &
{% for f_kind in datatypes %}
{%   for rank in range(0,max_rank+1) %}
{%     set routine_name = name+"_" + f_kind + "_" + rank|string %}
    {{routine_name}}{{",&\n" if not loop.last}}{% endfor %}{{ ",&" if not loop.last}}
{% endfor %}
end interface
{% endmacro %}
{########################################################################################}
{% macro render_print_ivar(ivar) %}
{#########################}
{# name and Fortran type #}
{#########################}
! {{ivar.name}}
write( output_unit, "(a)", advance="no") "{{ivar.name}}"
write( output_unit, "(a)", advance="no") ","
write( output_unit, "(a)", advance="no") "{{ivar.f_type}}"
{#######}
{# len #}
{#######}
write( output_unit, "(a)", advance="no") ","
{% if ivar.f_type == "character" %}
write( output_unit, "(i0)", advance="no") len({{ivar.name}})
{% else %}
write( output_unit, "(a)", advance="no") "<none>"
{% endif %}
{########}
{# kind #}
{########}
write( output_unit, "(a)", advance="no") ","
{% if ivar.f_type == "character" %}
if ( kind({{ivar.name}}) .eq. c_char ) then
  write( output_unit, "(a)", advance="no") "c_char"
else
  write( output_unit, "(a)", advance="no") kind({{ivar.name}})
endif
{% elif ivar.f_type == "type" and ivar.params|length %}
  write( output_unit, "(a)", advance="no") "{{ivar.f_kind}}({{ivar.params|join(",")}})"
{% elif ivar.f_type == "type" %}
  write( output_unit, "(a)", advance="no") "{{ivar.f_kind}}"
{% elif ivar.f_type == "logical" %}
if ( kind({{ivar.name}}) .eq. c_bool ) then
  write( output_unit, "(a)", advance="no") "c_bool"
else
  write( output_unit, "(a)", advance="no") kind({{ivar.name}})
endif
{% elif ivar.f_type == "integer" %}
if ( kind({{ivar.name}}) .eq. c_short ) then
  write( output_unit, "(a)", advance="no") "c_short"
else if ( kind({{ivar.name}}) .eq. c_int ) then
  write( output_unit, "(a)", advance="no") "c_int"
else if ( kind({{ivar.name}}) .eq. c_long ) then
  write( output_unit, "(a)", advance="no") "c_long"
else
  write( output_unit, "(a)", advance="no") kind({{ivar.name}})
endif
{% elif ivar.f_type == "real" %}
if ( kind({{ivar.name}}) .eq. c_float ) then
  write( output_unit, "(a)", advance="no") "c_float"
else if ( kind({{ivar.name}}) .eq. c_double ) then
  write( output_unit, "(a)", advance="no") "c_double"
else
  write( output_unit, "(a)", advance="no") kind({{ivar.name}})
endif
{% elif ivar.f_type == "complex" %}
if ( kind({{ivar.name}}) .eq. c_float_complex ) then
  write( output_unit, "(a)", advance="no") "c_float_complex"
else if ( kind({{ivar.name}}) .eq. c_double_complex ) then
  write( output_unit, "(a)", advance="no") "c_double_complex"
else
  write( output_unit, "(a)", advance="no") kind({{ivar.name}})
endif
{% else %}
write( output_unit, "(a)", advance="no") kind({{ivar.name}})
{% endif %}
{#####################}
{# bytes per element #}
{#####################}
write( output_unit, "(a)", advance="no") ","
write( output_unit, "(i0)", advance="no") c_sizeof({{ivar.name}})
{########}
{# rank #}
{########}
write( output_unit, "(a)", advance="no") ","
{% if ivar.rank > 0 %}
write( output_unit, "(i0)", advance="no") size(shape({{ivar.name}}))
{% else %}
write( output_unit, "(i0)", advance="no") 0
{% endif %}
{###########}
{# extents #}
{###########}
write( output_unit, "(a)", advance="no") ","
{% if ivar.rank > 0 %}
write( output_unit, "(a)", advance="no") "("
do i = 1,size(shape({{ivar.name}}))
  if ( i .ne. 1 ) write( output_unit, "(a)", advance="no") ","
  write( output_unit, "(a)", advance="no") size({{ivar.name}},i)
end do
write( output_unit, "(a)", advance="no") ")"
{% else %}
write( output_unit, "(a)", advance="no") "()"
{% endif %}
{################}
{# lower bounds #}
{################}
write( output_unit, "(a)", advance="no") ","
{% if ivar.rank > 0 %}
write( output_unit, "(a)", advance="no") "("
do i = 1,size(shape({{ivar.name}}))
  if ( i .ne. 1 ) write( output_unit, "(a)", advance="no") ","
  write( output_unit, "(i0)", advance="no") lbound({{ivar.name}},i)
end do
write( output_unit, "(a)", advance="no") ")"
{% else %}
write( output_unit, "(a)", advance="no") "()"
{% endif %}
{###################}
{# right-hand side #}
{###################}
write( output_unit, "(a)", advance="no") ","
{% if ivar.f_type == "character" %}
write( output_unit, "(a)", advance="no") {{ivar.name}}
{% elif ivar.f_type == "type" %}
  write( output_unit, "(a)", advance="no") {{ivar.name}}
{% elif ivar.f_type == "logical" %}
if ( {{ivar.name}} ) then
  write( output_unit, "(a)", advance="no") "true"
else
  write( output_unit, "(a)", advance="no") "false"
endif
{% elif ivar.f_type == "integer" %}
write( output_unit, "(i0)", advance="no") {{ivar.name}}
{% elif ivar.f_type == "real" %}
if ( kind({{ivar.name}}) .eq. c_float ) then
  call print_value({{ivar.name}}) 
else if ( kind({{ivar.name}}) .eq. c_double ) then
  call print_value({{ivar.name}}) 
else
  write( output_unit, "(e23.15e3)", advance="no") {{ivar.name}}
endif
{% elif ivar.f_type == "complex" %}
if ( kind({{ivar.name}}) .eq. c_float_complex ) then
  call print_value({{ivar.name}}) 
else if ( kind({{ivar.name}}) .eq. c_double_complex ) then
  call print_value({{ivar.name}}) 
else
  write( output_unit, "(e23.15e3,a,e23.15e3)", advance="no") real({{ivar.name}},kind=c_double),",",dimag({{ivar.name}})
endif
{% else %}
write( output_unit, "(a)", advance="no") kind({{ivar.name}})
{% endif %}
{# newline #}
write( output_unit, "(a)", advance="yes") ""
{% endmacro %}
{########################################################################################}
{% macro render_compiler_information() %}
write( output_unit, '(2a)') &
'info:compiler version: ', compiler_version()
write( output_unit, '(2a)') &
'info:compiler options: ', compiler_options()
{% endmacro %}
{########################################################################################}
{% macro render_print_value_routines_scalars() %}
subroutine print_value_c_bool_0(val)
  use iso_c_binding
  use iso_fortran_env
  implicit none
  logical(c_bool) :: val
  if ( val ) then
    write( output_unit, "(a)", advance="no") "true"
  else
    write( output_unit, "(a)", advance="no") "false"
  endif
end subroutine

subroutine print_value_c_short_0(val)
  use iso_c_binding
  use iso_fortran_env
  implicit none
  integer(c_short) :: val
  write( output_unit, "(i0)", advance="no") val
end subroutine

subroutine print_value_c_int_0(val)
  use iso_c_binding
  use iso_fortran_env
  implicit none
  integer(c_int) :: val
  write( output_unit, "(i0)", advance="no") val
end subroutine

subroutine print_value_c_long_0(val)
  use iso_c_binding
  use iso_fortran_env
  implicit none
  integer(c_long) :: val
  write( output_unit, "(i0)", advance="no") val
end subroutine

subroutine print_value_c_float_0(val)
  use iso_c_binding
  use iso_fortran_env
  implicit none
  real(c_float) :: val
  write( output_unit, "(e13.6e2)", advance="no") val
end subroutine

subroutine print_value_c_double_0(val)
  use iso_c_binding
  use iso_fortran_env
  implicit none
  real(c_double) :: val
  write( output_unit, "(e23.15e3)", advance="no") val
end subroutine

subroutine print_value_c_float_complex_0(val)
  use iso_c_binding
  use iso_fortran_env
  implicit none
  complex(c_float_complex) :: val
  write( output_unit, "(e13.6e2,a,e13.6e2)", advance="no") real(val,kind=c_float),",",aimag(val)
end subroutine

subroutine print_value_c_double_complex_0(val)
  use iso_c_binding
  use iso_fortran_env
  implicit none
  complex(c_double_complex) :: val
  write( output_unit, "(e23.15e3,a,e23.15e3)", advance="no") real(val,kind=c_double),",",dimag(val)
end subroutine
{% endmacro %}
{########################################################################################}
{% macro render_resolve_scope_program(modules,program,ivars_to_resolve,max_rank=0) %}
{% set datatypes = [
   "c_bool",
   "c_short",
   "c_int",
   "c_long",
   "c_float",
   "c_double",
   "c_float_complex",
   "c_double_complex",]
%}
! This file was generated by GPUFORT 
{% for module in modules %}
module {{module.name}}
{{ cm.render_used_modules(module.used_modules) | indent(2,True) }}
  implicit none ! TODO more care about implicit statement in non-modules
{{ module.accessibility|indent(2,True) }}
{% if module.public|length %}
  public :: &
    {{ module.public|join(",&\n    ") }}
{% endif %}
{% if module.private|length %}
  private :: &
    {{ module.private|join(",&\n    ") }}
{% endif %}
{% for line in module.declarations %}
{{line|indent(2,True)}}
{% endfor %}
end module{{"\n" if not loop.last}}
{% endfor %}

program {{program.name}}
{{ cm.render_used_modules(program.used_modules) | indent(2,True) }}
  implicit none ! TODO more care about implicit statement in non-modules
{{ render_interface("print_value",datatypes,max_rank) | indent(2,True) }}
{% for line in program.declarations %}
{{line|indent(2,True)}}
{% endfor %}

{% for ivar in ivars_to_resolve %}
{{ render_print_ivar(ivar)|indent(2,True) }}
{% endfor %}
contains
{{ render_print_value_routines_scalars() | indent(2,True) }}
end program
{% endmacro %}
{########################################################################################}
