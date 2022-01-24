{#- SPDX-License-Identifier: MIT                                        -#}
{#- Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.-#}
{########################################################################################}
{%- macro render_param_decl(ivar,
                            qualifiers="",
                            array_and_type_suffix="",
                            interop_suffix="_interop") -%}
{%- if ivar.f_type in ["integer","real","logical","type"] -%}
{%-   if ivar.rank > 0 -%}
type(gpufort_array{{ivar.rank}}){{qualifiers}} :: {{ivar.name}}
{%-   else -%}
{%-     if ivar.f_type == "type" -%}
{{ivar.f_type}}({{ivar.kind}}{{interop_suffix}}){{qualifiers}} :: {{ivar.name}}
{%-     elif ivar.kind is not none and ivar.kind|length -%}
{{ivar.f_type}}({{ivar.kind}}){{qualifiers}} :: {{ivar.name}}
{%-     else -%}
{{ivar.f_type}}{{qualifiers}} :: {{ivar.name}}
{%-     endif -%}
{%-   endif -%}
{%- endif -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_param_decls(ivars,
                             qualifiers="",
                             array_and_type_suffix="",
                             interop_suffix="_interop",
                             sep="\n") -%}
{%- for ivar in ivars -%}{{render_param_decl(ivar,qualifiers,array_and_type_suffix,interop_suffix)}}{{sep if not loop.last}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_params(ivars,
                        sep=",") -%}
{%- for ivar in ivars -%}{{ivar.name}}{{sep if not loop.last}}{% endfor -%}
{%- endmacro -%}
{########################################################################################}
{%- macro render_derived_types(derived_types,
                               interop_suffix="_interop") -%}
{# TODO consider extend #}
{% for derived_type in derived_types %}
type, bind(c) :: {{derived_type.name}}{{interop_suffix}}
{{ (render_param_decls(derived_type.variables)) | indent(2,True) }}
end type {{derived_type.name}}{{interop_suffix}}
{% endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_derived_type_copy_scalars(derived_type,
                                           derived_types,
                                           inline=False,
                                           interop_suffix="_interop",
                                           orig_var="orig_type",
                                           interop_var="interop_type") -%}
{% for ivar in derived_type.variables %}
{%   if ivar.rank == 0 %}
{%     if ivar.f_type in ["integer","real","logical"] %}
{{interop_var}}%{{ivar.name}} = {{orig_var}}%{{ivar.name }}
{%     elif ivar.f_type == "type" %}
{%       if inline %}
{%         for derived_type2 in derived_types %}
{%           if derived_type2.name == ivar.kind %}
{{ render_derived_type_copy_scalars( derived_type2, derived_types,
     True, interop_suffix, orig_var+"%"+ivar.name, interop_var+"%"+ivar.name ) }}
{%           endif %}
{%         endfor %}
{%       else %}
call {{ivar.kind}}{{interop_suffix}}_copy_scalars({{interop_var}}%{{ivar.name}},{{orig_var}}%{{ivar.name}})  
{%       endif %}
{%     endif %}
{%   endif %}
{% endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_used_modules(used_modules) -%}
{% for module in used_modules %}
use {{module.name}}{{(", only: "+module.only|join(",") if module.only|length>0)}}
{% endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_derived_type_copy_scalars_routines(derived_types,
                                                    interop_suffix="_interop",
                                                    used_modules=[]) -%}
{% for derived_type in derived_types %}
subroutine {{derived_type.name}}{{interop_suffix}}_copy_scalars(interop_type,orig_type)
{% if used_modules|length %}
{{ render_used_modules(used_modules) }}
{% endif %}
  type({{derived_type.name}}{{interop_suffix}}),intent(inout) :: interop_type
  type({{derived_type.name}}),intent(in) :: orig_type
  !
  {{ render_derived_type_copy_scalars(derived_type,derived_types,False,interop_suffix) }}
end subroutine{{"\n" if not loop.last}}
{% endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_derived_type_size_bytes_routines(derived_types,
                                                  interop_suffix="_interop",
                                                  used_modules=[]) -%}
{% for derived_type in derived_types %} 
function {{derived_type.name}}{{interop_suffix}}_size_bytes() result(size_bytes)
{% if used_modules|length %}
{{ render_used_modules(used_modules) }}
{% endif %}
  type({{derived_type.name}}{{interop_suffix}}) :: dummy
  integer(c_int) :: size_bytes
  !
  size_bytes = int(c_sizeof(dummy),c_int)
end function
{% endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_derived_type_copy_array_member(derived_type,
                                                member,
                                                derived_types,
                                                sync_mode="gpufort_array_sync_copy",
                                                interop_suffix="_interop", 
                                                orig_var="orig_type",
                                                interop_var="interop_type") -%}
{% for ivar in derived_type.variables %}
{%   if ivar.name == member %}
{%     if ivar.f_type in ["integer","real","logical"] %}
call hipCheck(gpufort_array_init({{interop_var}}%{{member}},{{orig_var}}%{{member}},&
  alloc_mode=gpufort_array_wrap_host_alloc_device,&
  sync_mode={{sync_mode}}))
{%     elif ivar.f_type == "type" %}
call hipCheck(gpufort_array_init({{interop_var}}%{{member}},&
  {{ivar.kind}}{{interop_suffix}}_size_bytes(),&
  shape({{orig_var}}%{{member}}),&
  lbounds=lbound({{orig_var}}%{{member}}),&
  alloc_mode=gpufort_array_alloc_pinned_host_alloc_device)) 
  ! 
call c_f_pointer({{interop_var}}%{{member}}%data%data_host,&
  {{interop_var}}_{{member}},shape=shape({{orig_var}}%{{member}}))
!call hipCheck(gpufort_array_copy_to_device(mesh_interop%x))
{%     endif %}
{%   endif %}
{% endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_derived_type_copy_array_member_routines(derived_types,
                                                         used_modules=[],
                                                         interop_suffix="_interop", 
                                                         orig_var="orig_type",
                                                         interop_var="interop_type") -%}
{% for derived_type in derived_types %}
{%   for ivar in derived_type.variables %}
{%     if ivar.rank > 0 %} 
subroutine {{derived_type.name}}{{interop_suffix}}_copy_{{ivar.name}}({{interop_var}},{{orig_var}}_ptr,sync_mode)
{% if used_modules|length %}
{{ render_used_modules(used_modules) }}
{% endif %}
  type({{derived_type.name}}{{interop_suffix}}),intent(inout) :: {{interop_var}}
  type({{derived_type.name}},intent(in) :: {{orig_var}}
  !
  integer(kind(gpufort_array_sync_none)),intent(in) :: sync_mode
  !
{{ render_derived_type_copy_array_member(derived_type, ivar.name,
                                         derived_types, "sync_mode") | indent(2,True) }}
end subroutine{{"\n" if not loop.last}}
{%     endif %}
{%   endfor %}
{% endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_launcher(kernel,
                          launcher) -%}
{%- set num_global_vars = kernel.global_vars|length + kernel.global_reduced_vars|length -%}
function {{launcher.name}}(&
{% if launcher.kind in ["hip"] %}
    grid, block,&
{% endif %}
    sharedmem,stream{{"," if num_global_vars > 0 }}&
{{render_params(kernel.global_vars+kernel.global_reduced_vars,sep=",&\n") | indent(4,True)}}) bind(c,name="{{launcher.name}}_") &
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
{{ render_param_decls(kernel.global_vars+kernel.global_reduced_vars) | indent(2,True) }}
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
{{render_params(kernel.global_vars+kernel.global_reduced_vars,sep=",&\n") | indent(4,True)}}) bind(c,name="{{kernel.name}}_cpu") &
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
{{ render_param_decls(kernel.global_vars+kernel.global_reduced_vars,"","_in") | indent(2,True) }}
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
