{# SPDX-License-Identifier: MIT #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{########################################################################################}
{%- macro render_set_fptr_lower_bound(fptr,
                                     array,
                                     rank) -%}
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

#include "gpufort_acc_runtime.def"

module gpufort_acc_runtime
  use gpufort_acc_runtime_base
    
  !> Lookup device pointer for given host pointer.
  !> \param[in] condition condition that must be met, otherwise host pointer is returned. Defaults to '.true.'.
  !> \param[in] if_present Only return device pointer if one could be found for the host pointer.
  !>                       otherwise host pointer is returned. Defaults to '.false.'.
  !> \note Returns a c_null_ptr if the host pointer is invalid, i.e. not C associated.
{{ render_interface("gpufort_acc_use_device",True) | indent(2,True) }}

  !> copyin( list ) parallel, kernels, serial, data, enter data,
  !> declare
  !> When entering the region or at an enter data directive,
  !> if the data in list is already present on the current device, the
  !> appropriate reference count is incremented and that copy is
  !> used. Otherwise, it allocates device memory and copies the
  !> values from the encountering thread and sets the appropriate
  !> reference count to one. When exiting the region the structured
  !> reference count is decremented. If both reference counts are
  !> zero, the device memory is deallocated.
{{ render_interface("gpufort_acc_copyin",True) | indent(2,True) }}

  !> copyout( list ) parallel, kernels, serial, data, exit data,
  !> declare
  !> When entering the region, if the data in list is already present on
  !> the current device, the structured reference count is incremented
  !> and that copy is used. Otherwise, it allocates device memory and
  !> sets the structured reference count to one. At an exit data
  !> directive with no finalize clause or when exiting the region,
  !> the appropriate reference count is decremented. At an exit
  !> data directive with a finalize clause, the dynamic reference
  !> count is set to zero. In any case, if both reference counts are zero,
  !> the data is copied from device memory to the encountering
  !> thread and the device memory is deallocated.
{{ render_interface("gpufort_acc_copyout",True) | indent(2,True) }}

  !> copy( list ) parallel, kernels, serial, data, declare
  !> When entering the region, if the data in list is already present on
  !> the current device, the structured reference count is incremented
  !> and that copy is used. Otherwise, it allocates device memory
  !> and copies the values from the encountering thread and sets
  !> the structured reference count to one. When exiting the region,
  !> the structured reference count is decremented. If both reference
  !> counts are zero, the data is copied from device memory to the
  !> encountering thread and the device memory is deallocated.
{{ render_interface("gpufort_acc_copy",True) | indent(2,True) }}

  !> create( list ) parallel, kernels, serial, data, enter data,
  !> declare
  !> When entering the region or at an enter data directive,
  !> if the data in list is already present on the current device, the
  !> appropriate reference count is incremented and that copy
  !> is used. Otherwise, it allocates device memory and sets the
  !> appropriate reference count to one. When exiting the region,
  !> the structured reference count is decremented. If both reference
  !> counts are zero, the device memory is deallocated.
{{ render_interface("gpufort_acc_create",True) | indent(2,True) }}

  !> no_create( list ) parallel, kernels, serial, data
  !> When entering the region, if the data in list is already present on
  !> the current device, the structured reference count is incremented
  !> and that copy is used. Otherwise, no action is performed and any
  !> device code in the construct will use the local memory address
  !> for that data.
{{ render_interface("gpufort_acc_no_create",True) | indent(2,True) }}

  !> The delete clause may appear on exit data directives.
  !>
  !> For each var in varlist, if var is in shared memory, no action is taken; if var is not in shared memory,
  !> the delete clause behaves as follows:
  !> If var is not present in the current device memory, a runtime error is issued.
  !> Otherwise, the dynamic reference counter is updated:
  !> On an exit data directive with a finalize clause, the dynamic reference counter
  !> is set to zero.
  !> Otherwise, a present decrement action with the dynamic reference counter is performed.
  !> If var is a pointer reference, a detach action is performed. If both structured and dynamic
  !> reference counters are zero, a delete action is performed.
  !> An exit data directive with a delete clause and with or without a finalize clause is
  !> functionally equivalent to a call to
  !> the acc_delete_finalize or acc_delete API routine, respectively, as described in Section 3.2.23.
{{ render_interface("gpufort_acc_delete",True) | indent(2,True) }}

  !> present( list ) parallel, kernels, serial, data, declare
  !> When entering the region, the data must be present in device
  !> memory, and the structured reference count is incremented.
  !> When exiting the region, the structured reference count is
  !> decremented.
{{ render_interface("gpufort_acc_present",True) | indent(2,True) }}
{{ render_interface("gpufort_acc_present_or_copy",True) | indent(2,True) }}
{{ render_interface("gpufort_acc_present_or_copyin",True) | indent(2,True) }}
{{ render_interface("gpufort_acc_present_or_copyout",True) | indent(2,True) }}
{{ render_interface("gpufort_acc_present_or_create",True) | indent(2,True) }}

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
{{ render_interface("gpufort_acc_update_host",True) | indent(2,True) }}

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
{{ render_interface("gpufort_acc_update_device",True) | indent(2,True) }}

  contains

    !
    ! autogenerated routines for different inputs
    !

{% for tuple in datatypes -%}
{%- for dims in dimensions -%}
{% if dims > 0 %}
{% set shape = ',shape(hostptr)' %}
{% set size = 'size(hostptr)*' %}
{% set rank = ',dimension(' + ':,'*(dims-1) + ':)' %}
{% else %}
{% set shape = '' %}
{% set size = '' %}
{% set rank = '' %}
{% endif %}
{% set suffix = tuple[0] + "_" + dims|string %}
    function gpufort_acc_use_device_{{suffix}}(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_use_device_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(in) :: hostptr
      integer(c_int),dimension({{ dims }}),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      {{tuple[2]}},pointer{{ rank }} :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufort_acc_use_device_b(c_loc(hostptr),{{size}}{{tuple[1]}}_8,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr{{shape}})
{% if dims > 0 %}
      if ( present(lbounds) ) then
{{ render_set_fptr_lower_bound("resultptr","hostptr",dims) | indent(8,True) }}
      endif
{% endif %}
    end function 

    function gpufort_acc_present_{{suffix}}(hostptr,module_var,or,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b, gpufort_acc_event_undefined
      implicit none
      {{tuple[2]}},target{{ rank }},intent(in) :: hostptr
      logical,intent(in),optional  :: module_var
      integer(kind(gpufort_acc_event_undefined)),&
              intent(in),optional  :: or
      integer,intent(in),optional  :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),{{size}}{{tuple[1]}}_8,module_var,or,async)
    end function
    
{% for or_clause_type in ["copy","copyin","copyout","create"] %}
    function gpufort_acc_present_or_{{or_clause_type}}_{{suffix}}(hostptr,module_var,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b, gpufort_acc_event_undefined
      implicit none
      {{tuple[2]}},target{{ rank }},intent(in) :: hostptr
      logical,intent(in),optional  :: module_var
      integer,intent(in),optional  :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),{{size}}{{tuple[1]}}_8,module_var,&
                                        gpufort_acc_event_{{or_clause_type}},async)
    end function
{% endfor %}

    function gpufort_acc_create_{{suffix}}(hostptr,module_var) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(in) :: hostptr
      logical,intent(in),optional              :: module_var
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),{{size}}{{tuple[1]}}_8)
    end function

    function gpufort_acc_no_create_{{suffix}}(hostptr,module_var) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(in) :: hostptr
      logical,intent(in),optional              :: module_var
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_{{suffix}}(hostptr,finalize,module_var)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(in) :: hostptr
      logical,intent(in),optional              :: finalize, module_var
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize,module_var)
    end subroutine

    function gpufort_acc_copyin_{{suffix}}(hostptr,async,module_var) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: module_var
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),{{size}}{{tuple[1]}}_8,async,module_var)
    end function

    function gpufort_acc_copyout_{{suffix}}(hostptr,async,module_var) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(inout) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: module_var
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),{{size}}{{tuple[1]}}_8,async,module_var)
    end function

    function gpufort_acc_copy_{{suffix}}(hostptr,async,module_var) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(inout) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: module_var
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),{{size}}{{tuple[1]}}_8,async)
    end function

    subroutine gpufort_acc_update_host_{{suffix}}(hostptr,condition,if_present,async,module_var)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present, module_var
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async,module_var)
    end subroutine

    subroutine gpufort_acc_update_device_{{suffix}}(hostptr,condition,if_present,async,module_var)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present, module_var
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async,module_var)
    end subroutine

{% endfor %}
{% endfor %}

end module
