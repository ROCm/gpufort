{# SPDX-License-Identifier: MIT #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{########################################################################################}
{% import "gpufortrt_api.macros.f03" as gam %}
{########################################################################################}
! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
module gpufortrt
  use gpufortrt_api_core
    
  !> Lookup device pointer for given host pointer.
  !> \param[in] condition condition that must be met, otherwise host pointer is returned. Defaults to '.true.'.
  !> \param[in] if_present Only return device pointer if one could be found for the host pointer.
  !>                       otherwise host pointer is returned. Defaults to '.false.'.
  !> \note Returns a c_null_ptr if the host pointer is invalid, i.e. not C associated.
{{ gam.render_interface("gpufortrt_use_device",True) | indent(2,True) }}

{{ gam.render_map_interfaces(datatypes) | indent(2,True) }}

{{ gam.render_map_and_lookup_interfaces(datatypes) | indent(2,True) }}


  !> Update Directive
{{ gam.render_interface_v2("gpufortrt_update_self",datatypes) | indent(2,True) }}
{{ gam.render_interface_v2("gpufortrt_update_device",datatypes) | indent(2,True) }}

  contains

{{ render_basic_map_routine() | indent(2,True) }}
{{ render_specialized_map_routines(datatypes) | indent(2,True) }}

{{ render_basic_map_routine() | indent(2,True) }}
{{ render_specialized_map_routines(datatypes) | indent(2,True) }}

{{ render_map_routines(data_clauses,datatypes,dimensions) | indent(2,True) }}
end module

{#
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
      use gpufortrt_core, only: gpufortrt_use_device_b
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
#}
