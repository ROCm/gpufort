{#- SPDX-License-Identifier: MIT                                        -#}
{#- Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.-#}
{########################################################################################}
{% import "common.macros.f03" as cm %}
{########################################################################################}
{% macro render_derived_types(derived_types,
                              interop_suffix="_interop")  %}
{# todo: consider extend #}
{% if derived_types|length %}

{% for derived_type in derived_types %}
type, bind(c) :: {{derived_type.name}}{{interop_suffix}}
{{ cm.render_param_decls(derived_type.variables) | indent(2,True) }}
end type {{derived_type.name}}{{interop_suffix}}{{"\n" if not loop.last}}
{% endfor %}
{% endif %}
{% endmacro  %}
{########################################################################################}
{% macro render_derived_type_copy_scalars(derived_type,
                                           derived_types,
                                           inline=False,
                                           direction="in",
                                           error_check="hipCheck",
                                           interop_suffix="_interop",
                                           orig_var="orig_type",
                                           interop_var="interop_type")  %}
{% for ivar in derived_type.variables %}
{%  if ivar.rank == 0  %}
{%    if ivar.f_type in ["integer","real","logical"]  %}
{%      if direction == "in"  %}
{{interop_var}}%{{ivar.name}} = {{orig_var}}%{{ivar.name }}
{%       else %}
{{orig_var}}%{{ivar.name}} = {{interop_var}}%{{ivar.name }}
{%      endif  %}
{%    elif ivar.f_type == "type"  %}
{%      if inline  %}
{%        for derived_type2 in derived_types %}
{%          if derived_type2.name == ivar.kind  %}
{{ render_derived_type_copy_scalars(derived_type2,
                                    derived_types,
                                    True,
                                    direction,
                                    error_check,
                                    interop_suffix,
                                    orig_var+"%"+ivar.name,
                                    interop_var+"%"+ivar.name) -}}
{%          endif  %}
{%        endfor %}
{%       else %}
call {{ivar.kind}}{{interop_suffix}}_copy{{direction}}_scalars({{interop_var}}%{{ivar.name}},{{orig_var}}%{{ivar.name}})
{%       endif  %}
{%     endif  %}
{%   endif  %}
{% endfor %}
{% endmacro  %}
{########################################################################################}
{% macro render_derived_type_copy_scalars_routines(derived_types,
                                                    direction="to",
                                                    used_modules=[],
                                                    error_check="hipCheck",
                                                    interop_suffix="_interop",
                                                    orig_var="orig_type",
                                                    interop_var="interop_type")  %}
{% for derived_type in derived_types %}
!> Copy scalar fields from the original type to the interoperable type.
!> Store a pointer to the original type as type(c_ptr).
!> \param[inout] {{interop_var}} the interoperable type
!> \param[in]    {{orig_var}} the original (non-interoperable) type
subroutine {{derived_type.name}}{{interop_suffix}}_copy{{direction}}_scalars({{interop_var}},{{orig_var}})
{% if used_modules|length  %}
{{ cm.render_used_modules(used_modules) | indent(2,True) -}}
{% endif  %}
  implicit none
  type({{derived_type.name}}{{interop_suffix}}),intent(inout) :: {{interop_var}}
  type({{derived_type.name}}),target,intent(in) :: {{orig_var}}
  !
  {{ render_derived_type_copy_scalars(derived_type,
                                      derived_types,
                                      False,
                                      direction,
                                      interop_suffix) -}}
end subroutine{{"\n" if not loop.last}}
{% endfor %}
{% endmacro  %}
{########################################################################################}
{% macro render_derived_type_size_bytes_routines(derived_types,
                                                  used_modules=[],
                                                  interop_suffix="_interop")  %}
{% for derived_type in derived_types %}
!> :return: Size of type '{{derived_type.name}}{{interop_suffix}}' in bytes.
function {{derived_type.name}}{{interop_suffix}}_size_bytes() result(size_bytes)
  implicit none
  type({{derived_type.name}}{{interop_suffix}}) :: dummy
  integer(c_size_t) :: size_bytes
  !
  size_bytes = c_sizeof(dummy)
end function
{% endfor %}
{% endmacro  %}
{########################################################################################}
{% macro render_derived_type_copy_derived_type_array_member(interop_type_name,
                                                            member,
                                                            rank,
                                                            direction="in",
                                                            orig_var="orig_type",
                                                            interop_var="interop_type")  %}
{% set rank_ub=rank+1 %}
{% set index = "i"+range(1,rank_ub)|join(",i") %}
call c_f_pointer({{interop_var}}%{{member}}%data%data_host,&
  {{interop_var}}_{{member}},shape=shape({{orig_var}}%{{member}}))
{{ cm.render_set_fptr_lower_bound(interop_var+"_"+member,
                          orig_var+"%"+member,
                          rank) -}}
!
{% for i in range(1,rank_ub) %}
do i{{i}} = lbound({{orig_var}},{{i}}),ubound({{orig_var}},{{i}})
{% endfor %}
    call {{interop_type_name}}_copy{{direction}}_scalars({{interop_var}}_{{member}}({{index}}),{{orig_var}}%{{member}}({{index}}))
{% for i in range(1,rank_ub) %}
end do
{% endfor %}
{% endmacro %}
{########################################################################################}
{% macro render_derived_type_copy_array_member_routines(derived_types,
                                                                      direction="in",
                                                                      asyncr="_async",
                                                                      used_modules=[],
                                                                      error_check="hipCheck",
                                                                      synchronize_queue="hipStreamSynchronize",
                                                                      interop_suffix="_interop",
                                                                      orig_var="orig_type",
                                                                      interop_var="interop_type")  %}
{% set is_async = asyncr == "_async" %}
{% set src = "original type's array member" if direction == "in" else "interoperable type's array member"  %}
{% set dest = "original type's array member" if direction != "out" else "interoperable type's array member"  %}
{% set tofrom = "to" if direction=="in" else "from"  %}
{% for derived_type in derived_types %}
{%   set interop_type_name = derived_type.name + interop_suffix %}
{%  for ivar in derived_type.variables %}
{%    if ivar.rank > 0 and ivar.f_type == "type"  %}
!> Copies data from the {{src}} to the {{dest}}.
{%      if direction == "in"  %}
!> \param[in] to_device Copies the {{dest}}'s host data to the device after the copy from the original type.
{%       else %}
!> \param[in] from_device Copies the {{src}}'s device data to the host before the copy to the original type.
{%      endif  %}
{%      if is_async  %}
!> \param[in] queue Asynchronous queue, e.g. HIP/CUDA stream, to use for the device data copies between host and device.
!> \note Device to host copy is blocking as it needs to complete before the interoperable type's
!>       host data can be copied to the original type.
{%      endif  %}
subroutine {{interop_type_name}}_copy{{direction}}_{{ivar.name}}{{asyncr}}(&
  {{interop_var}},{{orig_var}},&
  {{tofrom}}_device{{",queue" if is_async}})
{%      if used_modules|length  %}
{{ cm.render_used_modules(used_modules) | indent(2,True) -}}
{%      endif  %}
  implicit none
  type({{interop_type_name}}),intent(inout) :: {{interop_var}}
  type({{derived_type.name}}),intent(in) :: {{orig_var}}
  logical,intent(in) :: {{tofrom}}_device
{%      if asyncr == "_async"  %}
  type(c_ptr),intent(in) :: queue
{%      endif  %}
{%      if ivar.f_type == "type"  %}
  !
  logical :: allocate_temporary_host_buffer
  type({{ivar.kind}}{{interop_suffix}}),pointer :: {{interop_var}}_{{ivar.name}}
{%      endif  %}
  !
  allocate_temporary_host_buffer = {{interop_var}}%{{ivar.name}}%alloc_mode == gpufort_array_wrap_host_alloc_device .and. &
                         .not. c_associated({{interop_var}}%{{ivar.name}}%data%data_host)
  if ( allocate_temporary_host_buffer ) then ! allocate host array
    call gpufort_array_allocate_temporary_host_buffer({{interop_var}}%{{ivar.name}}%data%data_host,pinned=True,0)  
  endif
{%      if direction != "in"  %}
  if (from_device) then
    call {{error_check}}(gpufort_array_copy_to_host{{asyncr}}({{interop_var}}%{{ivar.name}}{{",queue" if is_async}}))
{%        if is_async  %}
    call {{error_check}}({{synchronize_queue}}(queue)}}
{%        endif  %}
  endif
{%      endif  %}
  !
{{ render_derived_type_copy_derived_type_array_member(interop_type_name,
                                                      ivar.name,
                                                      ivar.rank,
                                                      direction,
                                                      orig_var,
                                                      interop_var) | indent(2,True) -}}
{%      if direction == "in"  %}
  !
  if (to_device) then
    call {{error_check}}(gpufort_array_copy_to_device{{asyncr}}({{interop_var}}%{{ivar.name}}{{",queue" if is_async}}))
  endif
  if ( allocate_temporary_host_buffer ) then ! allocate host array
    call {{error_check}}({{synchronize_queue}}(queue)}}
    call gpufort_array_deallocate_temporary_host_buffer({{interop_var}}%{{ivar.name}}%data%data_host,pinned=True,0)  
  endif
{%      endif  %}
end subroutine{{"\n" if not loop.last}}
{%    endif  %}
{%  endfor %}
{% endfor %}
{% endmacro  %}
{########################################################################################}
{% macro render_derived_type_init_array_member(derived_type,
                                                member,
                                                asyncr="_async",
                                                error_check="hipCheck",
                                                interop_suffix="_interop",
                                                orig_var="orig_type",
                                                interop_var="interop_type")  %}
{% set is_async = asyncr == "_async" %}
{% set interop_type_name = derived_type.name + interop_suffix %}
{% for ivar in derived_type.variables %}
{%  if ivar.name == member  %}
{%    if ivar.f_type in ["integer","real","logical"]  %}
call {{error_check}}(gpufort_array_init{{asyncr}}({{interop_var}}%{{member}},{{orig_var}}%{{member}},&
  {{"queue," if is_async}}alloc_mode=gpufort_array_wrap_host_alloc_device,&
  sync_mode=sync_mode))
{%     elif ivar.f_type == "type"  %}
call {{error_check}}(gpufort_array_init({{interop_var}}%{{member}},&
  int({{ivar.kind}}{{interop_suffix}}_size_bytes(),c_int),&
  shape({{orig_var}}%{{member}}),&
  lbounds=lbound({{orig_var}}%{{member}}),&
  alloc_mode=alloc_mode))
{{interop_var}}%{{member}}%sync_mode=sync_mode ! set post init to prevent unnecessary copy at init
!
if ( sync_mode .eq. gpufort_array_sync_copy .or.&
     sync_mode .eq. gpufort_array_sync_copyin) then
  call {{interop_type_name}}_copyin_{{member}}{{asyncr}}({{interop_var}},{{orig_var}},.true.{{",queue" if is_async}})
endif
{%     endif  %}
{%   endif  %}
{% endfor %}
{% endmacro  %}
{########################################################################################}
{% macro render_derived_type_init_array_member_routines(derived_types,
                                                        asyncr="_async",
                                                        used_modules=[],
                                                        error_check="hipCheck",
                                                        interop_suffix="_interop",
                                                        orig_var="orig_type",
                                                        interop_var="interop_type")  %}
{% set is_async = asyncr == "_async" %}
{% for derived_type in derived_types %}
{% set interop_type_name = derived_type.name + interop_suffix %}
{%  for ivar in derived_type.variables %}
{%    if ivar.rank > 0  %}
!> Initialize the interoperable type's array member {{ivar.name}}.
!> \param[inout] {{interop_var}} the interoperable type
!> \param[in]    {{orig_var}} the original (non-interoperable) type
{%      if is_async  %}
!> \param[in] queue Asynchronous queue, e.g. HIP/CUDA stream, to use for the device data copies between host and device.
{%      endif  %}
!> \param[in] sync_mode Synchronization mode, i.e. what copies to perform at init and destruction.
subroutine {{interop_type_name}}_init_{{ivar.name}}{{asyncr}}(&
    {{interop_var}},&
    {{orig_var}},&
    sync_mode,&
    keep_host_copy,&
    {{"queue," if is_async}})
{%      if used_modules|length  %}
{{ cm.render_used_modules(used_modules) | indent(2,True) -}}
{%      endif  %}
  implicit none
  type({{derived_type.name}}{{interop_suffix}}),intent(inout) :: {{interop_var}}
  type({{derived_type.name}}),intent(in) :: {{orig_var}}
  integer(kind(gpufort_array_sync_none)),intent(in) :: sync_mode
  logical,intent(in) :: keep_host_copy
{%      if is_async  %}
  type(c_ptr),intent(in) :: queue
{%      endif  %}
  !
  integer(kind(gpufort_array_sync_none)) :: alloc_mode
  !
  if ( keep_host_copy ) then
    alloc_mode = gpufort_array_alloc_pinned_host_alloc_device
  else 
    alloc_mode = gpufort_array_wrap_host_alloc_device
  endif 
  !
{{ render_derived_type_init_array_member(derived_type,
                                         ivar.name,
                                         asyncr,
                                         error_check,
                                         interop_suffix,
                                         orig_var,
                                         interop_var) | indent(2,True) -}}
end subroutine{{"\n" if not loop.last}}
{%    endif  %}
{%  endfor %}
{% endfor %}
{% endmacro  %}
{########################################################################################}
{% macro render_derived_type_destroy_derived_type_array_member(interop_type_name,
                                                                member,
                                                                interop_member_type_name,
                                                                rank,
                                                                asyncr="_async",
                                                                orig_var="orig_type",
                                                                interop_var="interop_type")  %}
{% set is_async = asyncr == "_async" %}
{% set rank_ub=rank+1 %}
{% set index = "i"+range(1,rank_ub)|join(",i") %}
call c_f_pointer({{interop_var}}%{{member}}%data%data_host,&
  {{interop_var}}_{{member}},shape=shape({{orig_var}}%{{member}}))
{{ cm.render_set_fptr_lower_bound(interop_var+"_"+member,
                               orig_var+"%"+member,
                               rank) -}}
!
{% for i in range(1,rank_ub)  %}
do i{{i}} = lbound({{orig_var}},{{i}}),ubound({{orig_var}},{{i}})
{% endfor %}
    call {{interop_member_type_name}}_destroy{{asyncr}}({{interop_var}}_{{member}}({{index}}),{{orig_var}}%{{member}}({{index}}){{",queue" if is_async}})
{% for i in range(1,rank_ub)  %}
end do
{% endfor %}
{% endmacro %}
{########################################################################################}
{% macro render_derived_type_destroy_array_member(derived_type,
                                                  member,
                                                  asyncr="_async",
                                                  synchronize_queue="hipStreamSynchronize",
                                                  error_check="hipCheck",
                                                  interop_suffix="_interop",
                                                  orig_var="orig_type",
                                                  interop_var="interop_type")  %}
{% set is_async = asyncr == "_async" %}
{% set interop_type_name = derived_type.name + interop_suffix %}
{% for ivar in derived_type.variables %}
{%  if ivar.name == member  %}
{%    if ivar.f_type == "type"  %}{# destroy is sufficient for arrays of basic datatypes #}
{{ render_derived_type_destroy_derived_type_array_member(interop_type_name,
                                                         member,
                                                         ivar.kind+interop_suffix,
                                                         ivar.rank,
                                                         asyncr="_async",
                                                         orig_var="orig_type",
                                                         interop_var="interop_type") -}}
!
if ( {{interop_var}}%{{member}}%sync_mode .eq. gpufort_array_sync_copy .or.&
     {{interop_var}}%{{member}}%sync_mode .eq. gpufort_array_sync_copyin) then
  call {{interop_type_name}}_copyout_{{member}}{{asyncr}}({{interop_var}},{{orig_var}},.true.{{",queue" if is_async}})
endif
!
{{interop_var}}%{{member}}%sync_mode=gpufort_array_sync_none ! set pre destruction to prevent unnecessary copy at destruction
{%    endif  %}
call {{error_check}}(gpufort_array_destroy{{asyncr}}({{interop_var}}%{{member}}{{",queue" if is_async}}))
{%-   endif  %}
{% endfor %}
{% endmacro  %}
{########################################################################################}
{% macro render_derived_type_destroy_array_member_routines(derived_types,
                                                            asyncr="_async",
                                                            used_modules=[],
                                                            synchronize_queue="hipStreamSynchronize",
                                                            error_check="hipCheck",
                                                            interop_suffix="_interop",
                                                            orig_var="orig_type",
                                                            interop_var="interop_type")  %}
{% set is_async = asyncr == "_async" %}
{% for derived_type in derived_types %}
{% set interop_type_name = derived_type.name + interop_suffix %}
{%  for ivar in derived_type.variables %}
{%    if ivar.rank > 0  %}
!> Destroys the interoperable type's member '{{ivar.name}}' and
!> performs the copyout operation specified by the sync_mode field
!> of the interoperable type.
!> \param[inout] {{interop_var}} the interoperable type
!> \param[inout] {{orig_var}} the original (non-interoperable) type
{%      if is_async  %}
!> \param[in] queue Asynchronous queue, e.g. HIP/CUDA stream, to use for the device data copy from device to host if specified
!>            by the sync mode field.
!> \note Device to host copy is blocking as it needs to complete before the interoperable type's
!>       host data can be copied to the original type.
{%      endif  %}
subroutine {{interop_type_name}}_destroy_{{ivar.name}}{{asyncr}}(&
    {{interop_var}},&
    {{orig_var}}{{",queue" if is_async}})
{% if used_modules|length  %}
{{ cm.render_used_modules(used_modules) | indent(2,True) -}}
{% endif  %}
  implicit none
  type({{interop_type_name}}),intent(inout) :: {{interop_var}}
  type({{derived_type.name}}),intent(inout) :: {{orig_var}}
{%  if is_async  %}
  type(c_ptr),intent(in) :: queue
{%  endif  %}
{%  if ivar.f_type == "type"  %}
  !
  type({{ivar.kind}}{{interop_suffix}}),pointer :: {{interop_var}}_{{ivar.name}}
{%  endif  %}
  !
{%  set member=ivar.name %}
  if ({{interop_var}}%{{member}}%bytes_per_element .gt. 0) then ! implies initialized
{{ render_derived_type_destroy_array_member(derived_type,
                                            member,
                                            asyncr,
                                            synchronize_queue,
                                            error_check,
                                            interop_suffix,
                                            orig_var,
                                            interop_var) | indent(4,True) }}
  endif
end subroutine{{"\n" if not loop.last}}
{%    endif  %}
{%  endfor %}
{% endfor %}
{% endmacro  %}
{########################################################################################}
{% macro render_derived_type_destroy_routines(derived_types,
                                               asyncr="_async",
                                               used_modules=[],
                                               interop_suffix="_interop",
                                               orig_var="orig_type",
                                               interop_var="interop_type")  %}
{% set is_async = asyncr == "_async" %}
{% for derived_type in derived_types %}
{%   set interop_type_name = derived_type.name + interop_suffix %}
!> Destroys the interoperable type's members and
!> performs the copyout operation specified by the sync_mode field
!> of the respective member.
!> \param[inout] {{interop_var}} the interoperable type
!> \param[inout] {{orig_var}} the original (non-interoperable) type
{%   if is_async  %}
!> \param[in] queue Asynchronous queue, e.g. HIP/CUDA stream, to use for the device data copy from device to host if specified
!>            by the sync mode field.
!> \note Device to host copy is blocking as it needs to complete before the interoperable type's
!>       host data can be copied to the original type.
{%  endif  %}
subroutine {{interop_type_name}}_destroy{{asyncr}}(&
    {{interop_var}},&
    {{orig_var}}{{",queue" if is_async}})
{%  if used_modules|length  %}
{{ cm.render_used_modules(used_modules) | indent(2,True) -}}
{%  endif  %}
  implicit none
  type({{interop_type_name}}),intent(inout) :: {{interop_var}}
  type({{derived_type.name}}),target,intent(inout) :: {{orig_var}}
{%  if is_async  %}
  type(c_ptr),intent(in) :: queue
{%  endif  %}
  !
{%  for ivar in derived_type.variables %}
{%     set member = ivar.name %}
{%    if ivar.rank > 0  %}
  call {{interop_type_name}}_destroy_{{member}}{{asyncr}}(&
    {{interop_var}},&
    {{orig_var}}{{",queue" if is_async}})
{%    elif ivar.f_type=="type"  %}
{%      set interop_member_type_name = ivar.kind+interop_suffix %}
  call {{interop_member_type_name}}_destroy{{asyncr}}(&
    {{interop_var}}%{{member}},&
    {{orig_var}}%{{member}}{{",queue" if is_async}})
{%    endif  %}
{%  endfor %}
end subroutine

!> Destroys the interoperable type's members and
!> performs the copyout operation specified by the sync_mode field
!> of the respective member.
!> \param[inout] {{interop_var}}_cptr C pointer to the interoperable type
!> \param[inout] {{orig_var}}_cptr C pointer to the original (non-interoperable) type
{%   if is_async  %}
!> \param[in] queue Asynchronous queue, e.g. HIP/CUDA stream, to use for the device data copy from device to host if specified
!>            by the sync mode field.
!> \note Device to host copy is blocking as it needs to complete before the interoperable type's
!>       host data can be copied to the original type.
{%  endif %}
subroutine {{interop_type_name}}_destroy{{asyncr}}_cptr(&
    {{interop_var}}_cptr,&
    {{orig_var}}_cptr{{",queue" if is_async}})
{%  if used_modules|length  %}
{{ cm.render_used_modules(used_modules) | indent(2,True) -}}
{%  endif  %}
  implicit none
  type(c_ptr),intent(inout) :: {{interop_var}}_cptr,{{orig_var}}_cptr
{%  if is_async  %}
  type(c_ptr),intent(in)    :: queue
{%  endif  %}
  !
  type({{interop_type_name}}),pointer :: {{interop_var}}
  type({{derived_type.name}}),pointer :: {{orig_var}}
  !
  call c_f_pointer({{interop_var}}_cptr,{{interop_var}})
  call c_f_pointer({{orig_var}}_cptr,{{orig_var}})
  call {{interop_type_name}}_destroy{{asyncr}}(&
    {{interop_var}},&
    {{orig_var}}{{",queue" if is_async}})
end subroutine{{"\n" if not loop.last}}
{% endfor %}
{% endmacro  %}
{########################################################################################}