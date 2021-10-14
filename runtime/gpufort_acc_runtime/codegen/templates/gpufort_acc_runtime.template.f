{# SPDX-License-Identifier: MIT #}
{# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved. #}
! SPDX-License-Identifier: MIT
! Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.

! TODO not thread-safe, make use of OMP_LIB module if this becomes a problem

! TODO will be slow for large pointer storages 
! If this becomes an issue, use faster data structures or bind the public interfaces to
! a faster C runtime

module gpufort_acc_runtime
  use gpufort_acc_runtime_base

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
  interface gpufort_acc_copyin
    module procedure gpufort_acc_copyin_b
{%- for tuple in datatypes -%}
{%- for dims in dimensions -%}
{%- set name = "gpufort_acc_copyin_" + tuple[0] + "_" + dims|string -%}                                                              
,{{name}} 
{%- endfor -%} 
{%- endfor %} 
  end interface
  
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
  interface gpufort_acc_copyout
    module procedure gpufort_acc_copyout_b
{%- for tuple in datatypes -%}
{%- for dims in dimensions -%}
{%- set name = "gpufort_acc_copyout_" + tuple[0] + "_" + dims|string -%}                                                              
,{{name}} 
{%- endfor -%} 
{%- endfor %} 
  end interface
    
  !> copy( list ) parallel, kernels, serial, data, declare
  !> When entering the region, if the data in list is already present on
  !> the current device, the structured reference count is incremented
  !> and that copy is used. Otherwise, it allocates device memory
  !> and copies the values from the encountering thread and sets
  !> the structured reference count to one. When exiting the region,
  !> the structured reference count is decremented. If both reference
  !> counts are zero, the data is copied from device memory to the
  !> encountering thread and the device memory is deallocated.
  interface gpufort_acc_copy
    module procedure gpufort_acc_copy_b
{%- for tuple in datatypes -%}
{%- for dims in dimensions -%}
{%- set name = "gpufort_acc_copy_" + tuple[0] + "_" + dims|string -%}                                                              
,{{name}} 
{%- endfor -%} 
{%- endfor %} 
  end interface

  !> create( list ) parallel, kernels, serial, data, enter data,
  !> declare
  !> When entering the region or at an enter data directive,
  !> if the data in list is already present on the current device, the
  !> appropriate reference count is incremented and that copy
  !> is used. Otherwise, it allocates device memory and sets the
  !> appropriate reference count to one. When exiting the region,
  !> the structured reference count is decremented. If both reference
  !> counts are zero, the device memory is deallocated.  
  interface gpufort_acc_create
    module procedure gpufort_acc_create_b
{%- for tuple in datatypes -%}
{%- for dims in dimensions -%}
{%- set name = "gpufort_acc_create_" + tuple[0] + "_" + dims|string -%}                                                              
,{{name}} 
{%- endfor -%} 
{%- endfor %} 
  end interface

  !> no_create( list ) parallel, kernels, serial, data
  !> When entering the region, if the data in list is already present on
  !> the current device, the structured reference count is incremented
  !> and that copy is used. Otherwise, no action is performed and any
  !> device code in the construct will use the local memory address
  !> for that data.  
  interface gpufort_acc_no_create
    module procedure gpufort_acc_no_create_b
{%- for tuple in datatypes -%}
{%- for dims in dimensions -%}
{%- set name = "gpufort_acc_no_create_" + tuple[0] + "_" + dims|string -%}                                                              
,{{name}} 
{%- endfor -%} 
{%- endfor %} 
  end interface
  
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
  !>
  !> \note use 
  interface gpufort_acc_delete
    module procedure gpufort_acc_delete_b
{%- for tuple in datatypes -%}
{%- for dims in dimensions -%}
{%- set name = "gpufort_acc_delete_" + tuple[0] + "_" + dims|string -%}                                                              
,{{name}} 
{%- endfor -%} 
{%- endfor %} 
  end interface
 
  !> present( list ) parallel, kernels, serial, data, declare
  !> When entering the region, the data must be present in device
  !> memory, and the structured reference count is incremented.
  !> When exiting the region, the structured reference count is
  !> decremented. 
  interface gpufort_acc_present
    module procedure gpufort_acc_present_b
{%- for tuple in datatypes -%}
{%- for dims in dimensions -%}
{%- set name = "gpufort_acc_present_" + tuple[0] + "_" + dims|string -%}                                                              
,{{name}} 
{%- endfor -%} 
{%- endfor %} 
  end interface
 
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
  interface gpufort_acc_update_host
    module procedure gpufort_acc_update_host_b
{%- for tuple in datatypes -%}
{%- for dims in dimensions -%}
{%- set name = "gpufort_acc_update_host_" + tuple[0] + "_" + dims|string -%}                                                              
,{{name}} 
{%- endfor -%} 
{%- endfor %} 
  end interface
  
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
  interface gpufort_acc_update_device
    module procedure gpufort_acc_update_device_b
{%- for tuple in datatypes -%}
{%- for dims in dimensions -%}
{%- set name = "gpufort_acc_update_device_" + tuple[0] + "_" + dims|string -%}                                                              
,{{name}} 
{%- endfor -%} 
{%- endfor %} 
  end interface

  contains
    
    !
    ! autogenerated routines for different inputs
    !

{% for tuple in datatypes -%}
{%- for dims in dimensions -%}
{% if dims > 0 %}
{% set size = 'size(hostptr)*' %}
{% set rank = ',dimension(' + ':,'*(dims-1) + ':)' %}
{% else %}
{% set size = '' %}
{% set rank = '' %}
{% endif %}
{% set suffix = tuple[0] + "_" + dims|string %}                                                              
    function gpufort_acc_present_{{suffix}}(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),{{size}}{{tuple[1]}}_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_{{suffix}}(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),{{size}}{{tuple[1]}}_8)
    end function

    function gpufort_acc_no_create_{{suffix}}(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_{{suffix}}(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_{{suffix}}(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),{{size}}{{tuple[1]}}_8,async)
    end function

    function gpufort_acc_copyout_{{suffix}}(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),{{size}}{{tuple[1]}}_8,async)
    end function

    function gpufort_acc_copy_{{suffix}}(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),{{size}}{{tuple[1]}}_8,async)
    end function

    subroutine gpufort_acc_update_host_{{suffix}}(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_{{suffix}}(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      {{tuple[2]}},target{{ rank }},intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

{% endfor %} 
{% endfor %} 

end module
