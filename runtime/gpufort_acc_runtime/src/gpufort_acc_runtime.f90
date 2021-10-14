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
    module procedure gpufort_acc_copyin_b,gpufort_acc_copyin_l_0,gpufort_acc_copyin_l_1,gpufort_acc_copyin_l_2,gpufort_acc_copyin_l_3,gpufort_acc_copyin_l_4,gpufort_acc_copyin_l_5,gpufort_acc_copyin_l_6,gpufort_acc_copyin_l_7,gpufort_acc_copyin_i4_0,gpufort_acc_copyin_i4_1,gpufort_acc_copyin_i4_2,gpufort_acc_copyin_i4_3,gpufort_acc_copyin_i4_4,gpufort_acc_copyin_i4_5,gpufort_acc_copyin_i4_6,gpufort_acc_copyin_i4_7,gpufort_acc_copyin_i8_0,gpufort_acc_copyin_i8_1,gpufort_acc_copyin_i8_2,gpufort_acc_copyin_i8_3,gpufort_acc_copyin_i8_4,gpufort_acc_copyin_i8_5,gpufort_acc_copyin_i8_6,gpufort_acc_copyin_i8_7,gpufort_acc_copyin_r4_0,gpufort_acc_copyin_r4_1,gpufort_acc_copyin_r4_2,gpufort_acc_copyin_r4_3,gpufort_acc_copyin_r4_4,gpufort_acc_copyin_r4_5,gpufort_acc_copyin_r4_6,gpufort_acc_copyin_r4_7,gpufort_acc_copyin_r8_0,gpufort_acc_copyin_r8_1,gpufort_acc_copyin_r8_2,gpufort_acc_copyin_r8_3,gpufort_acc_copyin_r8_4,gpufort_acc_copyin_r8_5,gpufort_acc_copyin_r8_6,gpufort_acc_copyin_r8_7,gpufort_acc_copyin_c4_0,gpufort_acc_copyin_c4_1,gpufort_acc_copyin_c4_2,gpufort_acc_copyin_c4_3,gpufort_acc_copyin_c4_4,gpufort_acc_copyin_c4_5,gpufort_acc_copyin_c4_6,gpufort_acc_copyin_c4_7,gpufort_acc_copyin_c8_0,gpufort_acc_copyin_c8_1,gpufort_acc_copyin_c8_2,gpufort_acc_copyin_c8_3,gpufort_acc_copyin_c8_4,gpufort_acc_copyin_c8_5,gpufort_acc_copyin_c8_6,gpufort_acc_copyin_c8_7 
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
    module procedure gpufort_acc_copyout_b,gpufort_acc_copyout_l_0,gpufort_acc_copyout_l_1,gpufort_acc_copyout_l_2,gpufort_acc_copyout_l_3,gpufort_acc_copyout_l_4,gpufort_acc_copyout_l_5,gpufort_acc_copyout_l_6,gpufort_acc_copyout_l_7,gpufort_acc_copyout_i4_0,gpufort_acc_copyout_i4_1,gpufort_acc_copyout_i4_2,gpufort_acc_copyout_i4_3,gpufort_acc_copyout_i4_4,gpufort_acc_copyout_i4_5,gpufort_acc_copyout_i4_6,gpufort_acc_copyout_i4_7,gpufort_acc_copyout_i8_0,gpufort_acc_copyout_i8_1,gpufort_acc_copyout_i8_2,gpufort_acc_copyout_i8_3,gpufort_acc_copyout_i8_4,gpufort_acc_copyout_i8_5,gpufort_acc_copyout_i8_6,gpufort_acc_copyout_i8_7,gpufort_acc_copyout_r4_0,gpufort_acc_copyout_r4_1,gpufort_acc_copyout_r4_2,gpufort_acc_copyout_r4_3,gpufort_acc_copyout_r4_4,gpufort_acc_copyout_r4_5,gpufort_acc_copyout_r4_6,gpufort_acc_copyout_r4_7,gpufort_acc_copyout_r8_0,gpufort_acc_copyout_r8_1,gpufort_acc_copyout_r8_2,gpufort_acc_copyout_r8_3,gpufort_acc_copyout_r8_4,gpufort_acc_copyout_r8_5,gpufort_acc_copyout_r8_6,gpufort_acc_copyout_r8_7,gpufort_acc_copyout_c4_0,gpufort_acc_copyout_c4_1,gpufort_acc_copyout_c4_2,gpufort_acc_copyout_c4_3,gpufort_acc_copyout_c4_4,gpufort_acc_copyout_c4_5,gpufort_acc_copyout_c4_6,gpufort_acc_copyout_c4_7,gpufort_acc_copyout_c8_0,gpufort_acc_copyout_c8_1,gpufort_acc_copyout_c8_2,gpufort_acc_copyout_c8_3,gpufort_acc_copyout_c8_4,gpufort_acc_copyout_c8_5,gpufort_acc_copyout_c8_6,gpufort_acc_copyout_c8_7 
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
    module procedure gpufort_acc_copy_b,gpufort_acc_copy_l_0,gpufort_acc_copy_l_1,gpufort_acc_copy_l_2,gpufort_acc_copy_l_3,gpufort_acc_copy_l_4,gpufort_acc_copy_l_5,gpufort_acc_copy_l_6,gpufort_acc_copy_l_7,gpufort_acc_copy_i4_0,gpufort_acc_copy_i4_1,gpufort_acc_copy_i4_2,gpufort_acc_copy_i4_3,gpufort_acc_copy_i4_4,gpufort_acc_copy_i4_5,gpufort_acc_copy_i4_6,gpufort_acc_copy_i4_7,gpufort_acc_copy_i8_0,gpufort_acc_copy_i8_1,gpufort_acc_copy_i8_2,gpufort_acc_copy_i8_3,gpufort_acc_copy_i8_4,gpufort_acc_copy_i8_5,gpufort_acc_copy_i8_6,gpufort_acc_copy_i8_7,gpufort_acc_copy_r4_0,gpufort_acc_copy_r4_1,gpufort_acc_copy_r4_2,gpufort_acc_copy_r4_3,gpufort_acc_copy_r4_4,gpufort_acc_copy_r4_5,gpufort_acc_copy_r4_6,gpufort_acc_copy_r4_7,gpufort_acc_copy_r8_0,gpufort_acc_copy_r8_1,gpufort_acc_copy_r8_2,gpufort_acc_copy_r8_3,gpufort_acc_copy_r8_4,gpufort_acc_copy_r8_5,gpufort_acc_copy_r8_6,gpufort_acc_copy_r8_7,gpufort_acc_copy_c4_0,gpufort_acc_copy_c4_1,gpufort_acc_copy_c4_2,gpufort_acc_copy_c4_3,gpufort_acc_copy_c4_4,gpufort_acc_copy_c4_5,gpufort_acc_copy_c4_6,gpufort_acc_copy_c4_7,gpufort_acc_copy_c8_0,gpufort_acc_copy_c8_1,gpufort_acc_copy_c8_2,gpufort_acc_copy_c8_3,gpufort_acc_copy_c8_4,gpufort_acc_copy_c8_5,gpufort_acc_copy_c8_6,gpufort_acc_copy_c8_7 
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
    module procedure gpufort_acc_create_b,gpufort_acc_create_l_0,gpufort_acc_create_l_1,gpufort_acc_create_l_2,gpufort_acc_create_l_3,gpufort_acc_create_l_4,gpufort_acc_create_l_5,gpufort_acc_create_l_6,gpufort_acc_create_l_7,gpufort_acc_create_i4_0,gpufort_acc_create_i4_1,gpufort_acc_create_i4_2,gpufort_acc_create_i4_3,gpufort_acc_create_i4_4,gpufort_acc_create_i4_5,gpufort_acc_create_i4_6,gpufort_acc_create_i4_7,gpufort_acc_create_i8_0,gpufort_acc_create_i8_1,gpufort_acc_create_i8_2,gpufort_acc_create_i8_3,gpufort_acc_create_i8_4,gpufort_acc_create_i8_5,gpufort_acc_create_i8_6,gpufort_acc_create_i8_7,gpufort_acc_create_r4_0,gpufort_acc_create_r4_1,gpufort_acc_create_r4_2,gpufort_acc_create_r4_3,gpufort_acc_create_r4_4,gpufort_acc_create_r4_5,gpufort_acc_create_r4_6,gpufort_acc_create_r4_7,gpufort_acc_create_r8_0,gpufort_acc_create_r8_1,gpufort_acc_create_r8_2,gpufort_acc_create_r8_3,gpufort_acc_create_r8_4,gpufort_acc_create_r8_5,gpufort_acc_create_r8_6,gpufort_acc_create_r8_7,gpufort_acc_create_c4_0,gpufort_acc_create_c4_1,gpufort_acc_create_c4_2,gpufort_acc_create_c4_3,gpufort_acc_create_c4_4,gpufort_acc_create_c4_5,gpufort_acc_create_c4_6,gpufort_acc_create_c4_7,gpufort_acc_create_c8_0,gpufort_acc_create_c8_1,gpufort_acc_create_c8_2,gpufort_acc_create_c8_3,gpufort_acc_create_c8_4,gpufort_acc_create_c8_5,gpufort_acc_create_c8_6,gpufort_acc_create_c8_7 
  end interface

  !> no_create( list ) parallel, kernels, serial, data
  !> When entering the region, if the data in list is already present on
  !> the current device, the structured reference count is incremented
  !> and that copy is used. Otherwise, no action is performed and any
  !> device code in the construct will use the local memory address
  !> for that data.  
  interface gpufort_acc_no_create
    module procedure gpufort_acc_no_create_b,gpufort_acc_no_create_l_0,gpufort_acc_no_create_l_1,gpufort_acc_no_create_l_2,gpufort_acc_no_create_l_3,gpufort_acc_no_create_l_4,gpufort_acc_no_create_l_5,gpufort_acc_no_create_l_6,gpufort_acc_no_create_l_7,gpufort_acc_no_create_i4_0,gpufort_acc_no_create_i4_1,gpufort_acc_no_create_i4_2,gpufort_acc_no_create_i4_3,gpufort_acc_no_create_i4_4,gpufort_acc_no_create_i4_5,gpufort_acc_no_create_i4_6,gpufort_acc_no_create_i4_7,gpufort_acc_no_create_i8_0,gpufort_acc_no_create_i8_1,gpufort_acc_no_create_i8_2,gpufort_acc_no_create_i8_3,gpufort_acc_no_create_i8_4,gpufort_acc_no_create_i8_5,gpufort_acc_no_create_i8_6,gpufort_acc_no_create_i8_7,gpufort_acc_no_create_r4_0,gpufort_acc_no_create_r4_1,gpufort_acc_no_create_r4_2,gpufort_acc_no_create_r4_3,gpufort_acc_no_create_r4_4,gpufort_acc_no_create_r4_5,gpufort_acc_no_create_r4_6,gpufort_acc_no_create_r4_7,gpufort_acc_no_create_r8_0,gpufort_acc_no_create_r8_1,gpufort_acc_no_create_r8_2,gpufort_acc_no_create_r8_3,gpufort_acc_no_create_r8_4,gpufort_acc_no_create_r8_5,gpufort_acc_no_create_r8_6,gpufort_acc_no_create_r8_7,gpufort_acc_no_create_c4_0,gpufort_acc_no_create_c4_1,gpufort_acc_no_create_c4_2,gpufort_acc_no_create_c4_3,gpufort_acc_no_create_c4_4,gpufort_acc_no_create_c4_5,gpufort_acc_no_create_c4_6,gpufort_acc_no_create_c4_7,gpufort_acc_no_create_c8_0,gpufort_acc_no_create_c8_1,gpufort_acc_no_create_c8_2,gpufort_acc_no_create_c8_3,gpufort_acc_no_create_c8_4,gpufort_acc_no_create_c8_5,gpufort_acc_no_create_c8_6,gpufort_acc_no_create_c8_7 
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
    module procedure gpufort_acc_delete_b,gpufort_acc_delete_l_0,gpufort_acc_delete_l_1,gpufort_acc_delete_l_2,gpufort_acc_delete_l_3,gpufort_acc_delete_l_4,gpufort_acc_delete_l_5,gpufort_acc_delete_l_6,gpufort_acc_delete_l_7,gpufort_acc_delete_i4_0,gpufort_acc_delete_i4_1,gpufort_acc_delete_i4_2,gpufort_acc_delete_i4_3,gpufort_acc_delete_i4_4,gpufort_acc_delete_i4_5,gpufort_acc_delete_i4_6,gpufort_acc_delete_i4_7,gpufort_acc_delete_i8_0,gpufort_acc_delete_i8_1,gpufort_acc_delete_i8_2,gpufort_acc_delete_i8_3,gpufort_acc_delete_i8_4,gpufort_acc_delete_i8_5,gpufort_acc_delete_i8_6,gpufort_acc_delete_i8_7,gpufort_acc_delete_r4_0,gpufort_acc_delete_r4_1,gpufort_acc_delete_r4_2,gpufort_acc_delete_r4_3,gpufort_acc_delete_r4_4,gpufort_acc_delete_r4_5,gpufort_acc_delete_r4_6,gpufort_acc_delete_r4_7,gpufort_acc_delete_r8_0,gpufort_acc_delete_r8_1,gpufort_acc_delete_r8_2,gpufort_acc_delete_r8_3,gpufort_acc_delete_r8_4,gpufort_acc_delete_r8_5,gpufort_acc_delete_r8_6,gpufort_acc_delete_r8_7,gpufort_acc_delete_c4_0,gpufort_acc_delete_c4_1,gpufort_acc_delete_c4_2,gpufort_acc_delete_c4_3,gpufort_acc_delete_c4_4,gpufort_acc_delete_c4_5,gpufort_acc_delete_c4_6,gpufort_acc_delete_c4_7,gpufort_acc_delete_c8_0,gpufort_acc_delete_c8_1,gpufort_acc_delete_c8_2,gpufort_acc_delete_c8_3,gpufort_acc_delete_c8_4,gpufort_acc_delete_c8_5,gpufort_acc_delete_c8_6,gpufort_acc_delete_c8_7 
  end interface
 
  !> present( list ) parallel, kernels, serial, data, declare
  !> When entering the region, the data must be present in device
  !> memory, and the structured reference count is incremented.
  !> When exiting the region, the structured reference count is
  !> decremented. 
  interface gpufort_acc_present
    module procedure gpufort_acc_present_b,gpufort_acc_present_l_0,gpufort_acc_present_l_1,gpufort_acc_present_l_2,gpufort_acc_present_l_3,gpufort_acc_present_l_4,gpufort_acc_present_l_5,gpufort_acc_present_l_6,gpufort_acc_present_l_7,gpufort_acc_present_i4_0,gpufort_acc_present_i4_1,gpufort_acc_present_i4_2,gpufort_acc_present_i4_3,gpufort_acc_present_i4_4,gpufort_acc_present_i4_5,gpufort_acc_present_i4_6,gpufort_acc_present_i4_7,gpufort_acc_present_i8_0,gpufort_acc_present_i8_1,gpufort_acc_present_i8_2,gpufort_acc_present_i8_3,gpufort_acc_present_i8_4,gpufort_acc_present_i8_5,gpufort_acc_present_i8_6,gpufort_acc_present_i8_7,gpufort_acc_present_r4_0,gpufort_acc_present_r4_1,gpufort_acc_present_r4_2,gpufort_acc_present_r4_3,gpufort_acc_present_r4_4,gpufort_acc_present_r4_5,gpufort_acc_present_r4_6,gpufort_acc_present_r4_7,gpufort_acc_present_r8_0,gpufort_acc_present_r8_1,gpufort_acc_present_r8_2,gpufort_acc_present_r8_3,gpufort_acc_present_r8_4,gpufort_acc_present_r8_5,gpufort_acc_present_r8_6,gpufort_acc_present_r8_7,gpufort_acc_present_c4_0,gpufort_acc_present_c4_1,gpufort_acc_present_c4_2,gpufort_acc_present_c4_3,gpufort_acc_present_c4_4,gpufort_acc_present_c4_5,gpufort_acc_present_c4_6,gpufort_acc_present_c4_7,gpufort_acc_present_c8_0,gpufort_acc_present_c8_1,gpufort_acc_present_c8_2,gpufort_acc_present_c8_3,gpufort_acc_present_c8_4,gpufort_acc_present_c8_5,gpufort_acc_present_c8_6,gpufort_acc_present_c8_7 
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
    module procedure gpufort_acc_update_host_b,gpufort_acc_update_host_l_0,gpufort_acc_update_host_l_1,gpufort_acc_update_host_l_2,gpufort_acc_update_host_l_3,gpufort_acc_update_host_l_4,gpufort_acc_update_host_l_5,gpufort_acc_update_host_l_6,gpufort_acc_update_host_l_7,gpufort_acc_update_host_i4_0,gpufort_acc_update_host_i4_1,gpufort_acc_update_host_i4_2,gpufort_acc_update_host_i4_3,gpufort_acc_update_host_i4_4,gpufort_acc_update_host_i4_5,gpufort_acc_update_host_i4_6,gpufort_acc_update_host_i4_7,gpufort_acc_update_host_i8_0,gpufort_acc_update_host_i8_1,gpufort_acc_update_host_i8_2,gpufort_acc_update_host_i8_3,gpufort_acc_update_host_i8_4,gpufort_acc_update_host_i8_5,gpufort_acc_update_host_i8_6,gpufort_acc_update_host_i8_7,gpufort_acc_update_host_r4_0,gpufort_acc_update_host_r4_1,gpufort_acc_update_host_r4_2,gpufort_acc_update_host_r4_3,gpufort_acc_update_host_r4_4,gpufort_acc_update_host_r4_5,gpufort_acc_update_host_r4_6,gpufort_acc_update_host_r4_7,gpufort_acc_update_host_r8_0,gpufort_acc_update_host_r8_1,gpufort_acc_update_host_r8_2,gpufort_acc_update_host_r8_3,gpufort_acc_update_host_r8_4,gpufort_acc_update_host_r8_5,gpufort_acc_update_host_r8_6,gpufort_acc_update_host_r8_7,gpufort_acc_update_host_c4_0,gpufort_acc_update_host_c4_1,gpufort_acc_update_host_c4_2,gpufort_acc_update_host_c4_3,gpufort_acc_update_host_c4_4,gpufort_acc_update_host_c4_5,gpufort_acc_update_host_c4_6,gpufort_acc_update_host_c4_7,gpufort_acc_update_host_c8_0,gpufort_acc_update_host_c8_1,gpufort_acc_update_host_c8_2,gpufort_acc_update_host_c8_3,gpufort_acc_update_host_c8_4,gpufort_acc_update_host_c8_5,gpufort_acc_update_host_c8_6,gpufort_acc_update_host_c8_7 
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
    module procedure gpufort_acc_update_device_b,gpufort_acc_update_device_l_0,gpufort_acc_update_device_l_1,gpufort_acc_update_device_l_2,gpufort_acc_update_device_l_3,gpufort_acc_update_device_l_4,gpufort_acc_update_device_l_5,gpufort_acc_update_device_l_6,gpufort_acc_update_device_l_7,gpufort_acc_update_device_i4_0,gpufort_acc_update_device_i4_1,gpufort_acc_update_device_i4_2,gpufort_acc_update_device_i4_3,gpufort_acc_update_device_i4_4,gpufort_acc_update_device_i4_5,gpufort_acc_update_device_i4_6,gpufort_acc_update_device_i4_7,gpufort_acc_update_device_i8_0,gpufort_acc_update_device_i8_1,gpufort_acc_update_device_i8_2,gpufort_acc_update_device_i8_3,gpufort_acc_update_device_i8_4,gpufort_acc_update_device_i8_5,gpufort_acc_update_device_i8_6,gpufort_acc_update_device_i8_7,gpufort_acc_update_device_r4_0,gpufort_acc_update_device_r4_1,gpufort_acc_update_device_r4_2,gpufort_acc_update_device_r4_3,gpufort_acc_update_device_r4_4,gpufort_acc_update_device_r4_5,gpufort_acc_update_device_r4_6,gpufort_acc_update_device_r4_7,gpufort_acc_update_device_r8_0,gpufort_acc_update_device_r8_1,gpufort_acc_update_device_r8_2,gpufort_acc_update_device_r8_3,gpufort_acc_update_device_r8_4,gpufort_acc_update_device_r8_5,gpufort_acc_update_device_r8_6,gpufort_acc_update_device_r8_7,gpufort_acc_update_device_c4_0,gpufort_acc_update_device_c4_1,gpufort_acc_update_device_c4_2,gpufort_acc_update_device_c4_3,gpufort_acc_update_device_c4_4,gpufort_acc_update_device_c4_5,gpufort_acc_update_device_c4_6,gpufort_acc_update_device_c4_7,gpufort_acc_update_device_c8_0,gpufort_acc_update_device_c8_1,gpufort_acc_update_device_c8_2,gpufort_acc_update_device_c8_3,gpufort_acc_update_device_c8_4,gpufort_acc_update_device_c8_5,gpufort_acc_update_device_c8_6,gpufort_acc_update_device_c8_7 
  end interface

  contains
    
    !
    ! autogenerated routines for different inputs
    !

                                                              
    function gpufort_acc_present_l_0(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      logical,target,intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),1_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_l_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      logical,target,intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),1_8)
    end function

    function gpufort_acc_no_create_l_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      logical,target,intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_l_0(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      logical,target,intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_l_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      logical,target,intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),1_8,async)
    end function

    function gpufort_acc_copyout_l_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      logical,target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),1_8,async)
    end function

    function gpufort_acc_copy_l_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      logical,target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),1_8,async)
    end function

    subroutine gpufort_acc_update_host_l_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      logical,target,intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_l_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      logical,target,intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_l_1(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      logical,target,dimension(:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*1_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_l_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      logical,target,dimension(:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*1_8)
    end function

    function gpufort_acc_no_create_l_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      logical,target,dimension(:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_l_1(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      logical,target,dimension(:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_l_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      logical,target,dimension(:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*1_8,async)
    end function

    function gpufort_acc_copyout_l_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      logical,target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*1_8,async)
    end function

    function gpufort_acc_copy_l_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      logical,target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*1_8,async)
    end function

    subroutine gpufort_acc_update_host_l_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      logical,target,dimension(:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_l_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      logical,target,dimension(:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_l_2(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      logical,target,dimension(:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*1_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_l_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      logical,target,dimension(:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*1_8)
    end function

    function gpufort_acc_no_create_l_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      logical,target,dimension(:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_l_2(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      logical,target,dimension(:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_l_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      logical,target,dimension(:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*1_8,async)
    end function

    function gpufort_acc_copyout_l_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      logical,target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*1_8,async)
    end function

    function gpufort_acc_copy_l_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      logical,target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*1_8,async)
    end function

    subroutine gpufort_acc_update_host_l_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      logical,target,dimension(:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_l_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      logical,target,dimension(:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_l_3(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      logical,target,dimension(:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*1_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_l_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      logical,target,dimension(:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*1_8)
    end function

    function gpufort_acc_no_create_l_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      logical,target,dimension(:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_l_3(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      logical,target,dimension(:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_l_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      logical,target,dimension(:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*1_8,async)
    end function

    function gpufort_acc_copyout_l_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      logical,target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*1_8,async)
    end function

    function gpufort_acc_copy_l_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      logical,target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*1_8,async)
    end function

    subroutine gpufort_acc_update_host_l_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      logical,target,dimension(:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_l_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      logical,target,dimension(:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_l_4(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      logical,target,dimension(:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*1_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_l_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      logical,target,dimension(:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*1_8)
    end function

    function gpufort_acc_no_create_l_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      logical,target,dimension(:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_l_4(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      logical,target,dimension(:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_l_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      logical,target,dimension(:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*1_8,async)
    end function

    function gpufort_acc_copyout_l_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      logical,target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*1_8,async)
    end function

    function gpufort_acc_copy_l_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      logical,target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*1_8,async)
    end function

    subroutine gpufort_acc_update_host_l_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      logical,target,dimension(:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_l_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      logical,target,dimension(:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_l_5(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      logical,target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*1_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_l_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      logical,target,dimension(:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*1_8)
    end function

    function gpufort_acc_no_create_l_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      logical,target,dimension(:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_l_5(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      logical,target,dimension(:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_l_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      logical,target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*1_8,async)
    end function

    function gpufort_acc_copyout_l_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      logical,target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*1_8,async)
    end function

    function gpufort_acc_copy_l_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      logical,target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*1_8,async)
    end function

    subroutine gpufort_acc_update_host_l_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      logical,target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_l_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      logical,target,dimension(:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_l_6(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*1_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_l_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*1_8)
    end function

    function gpufort_acc_no_create_l_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_l_6(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_l_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*1_8,async)
    end function

    function gpufort_acc_copyout_l_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*1_8,async)
    end function

    function gpufort_acc_copy_l_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*1_8,async)
    end function

    subroutine gpufort_acc_update_host_l_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_l_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_l_7(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*1_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_l_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*1_8)
    end function

    function gpufort_acc_no_create_l_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_l_7(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_l_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*1_8,async)
    end function

    function gpufort_acc_copyout_l_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*1_8,async)
    end function

    function gpufort_acc_copy_l_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*1_8,async)
    end function

    subroutine gpufort_acc_update_host_l_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_l_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

 
                                                              
    function gpufort_acc_present_i4_0(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      integer(4),target,intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),4_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_i4_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      integer(4),target,intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),4_8)
    end function

    function gpufort_acc_no_create_i4_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      integer(4),target,intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_i4_0(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      integer(4),target,intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_i4_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      integer(4),target,intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),4_8,async)
    end function

    function gpufort_acc_copyout_i4_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      integer(4),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),4_8,async)
    end function

    function gpufort_acc_copy_i4_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      integer(4),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),4_8,async)
    end function

    subroutine gpufort_acc_update_host_i4_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      integer(4),target,intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_i4_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      integer(4),target,intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_i4_1(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      integer(4),target,dimension(:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*4_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_i4_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      integer(4),target,dimension(:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*4_8)
    end function

    function gpufort_acc_no_create_i4_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      integer(4),target,dimension(:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_i4_1(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      integer(4),target,dimension(:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_i4_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      integer(4),target,dimension(:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copyout_i4_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      integer(4),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copy_i4_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      integer(4),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    subroutine gpufort_acc_update_host_i4_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      integer(4),target,dimension(:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_i4_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      integer(4),target,dimension(:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_i4_2(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      integer(4),target,dimension(:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*4_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_i4_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      integer(4),target,dimension(:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*4_8)
    end function

    function gpufort_acc_no_create_i4_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      integer(4),target,dimension(:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_i4_2(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      integer(4),target,dimension(:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_i4_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      integer(4),target,dimension(:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copyout_i4_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      integer(4),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copy_i4_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      integer(4),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    subroutine gpufort_acc_update_host_i4_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      integer(4),target,dimension(:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_i4_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      integer(4),target,dimension(:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_i4_3(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      integer(4),target,dimension(:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*4_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_i4_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      integer(4),target,dimension(:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*4_8)
    end function

    function gpufort_acc_no_create_i4_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      integer(4),target,dimension(:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_i4_3(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      integer(4),target,dimension(:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_i4_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      integer(4),target,dimension(:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copyout_i4_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      integer(4),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copy_i4_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      integer(4),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    subroutine gpufort_acc_update_host_i4_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      integer(4),target,dimension(:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_i4_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      integer(4),target,dimension(:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_i4_4(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      integer(4),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*4_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_i4_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      integer(4),target,dimension(:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*4_8)
    end function

    function gpufort_acc_no_create_i4_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      integer(4),target,dimension(:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_i4_4(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      integer(4),target,dimension(:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_i4_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      integer(4),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copyout_i4_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      integer(4),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copy_i4_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      integer(4),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    subroutine gpufort_acc_update_host_i4_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      integer(4),target,dimension(:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_i4_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      integer(4),target,dimension(:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_i4_5(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*4_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_i4_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*4_8)
    end function

    function gpufort_acc_no_create_i4_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_i4_5(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_i4_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copyout_i4_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copy_i4_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    subroutine gpufort_acc_update_host_i4_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_i4_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_i4_6(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*4_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_i4_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*4_8)
    end function

    function gpufort_acc_no_create_i4_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_i4_6(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_i4_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copyout_i4_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copy_i4_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    subroutine gpufort_acc_update_host_i4_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_i4_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_i4_7(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*4_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_i4_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*4_8)
    end function

    function gpufort_acc_no_create_i4_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_i4_7(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_i4_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copyout_i4_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copy_i4_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    subroutine gpufort_acc_update_host_i4_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_i4_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      integer(4),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

 
                                                              
    function gpufort_acc_present_i8_0(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      integer(8),target,intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),8_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_i8_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      integer(8),target,intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),8_8)
    end function

    function gpufort_acc_no_create_i8_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      integer(8),target,intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_i8_0(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      integer(8),target,intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_i8_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      integer(8),target,intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),8_8,async)
    end function

    function gpufort_acc_copyout_i8_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      integer(8),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),8_8,async)
    end function

    function gpufort_acc_copy_i8_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      integer(8),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),8_8,async)
    end function

    subroutine gpufort_acc_update_host_i8_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      integer(8),target,intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_i8_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      integer(8),target,intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_i8_1(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      integer(8),target,dimension(:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*8_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_i8_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      integer(8),target,dimension(:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*8_8)
    end function

    function gpufort_acc_no_create_i8_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      integer(8),target,dimension(:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_i8_1(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      integer(8),target,dimension(:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_i8_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      integer(8),target,dimension(:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copyout_i8_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      integer(8),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copy_i8_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      integer(8),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    subroutine gpufort_acc_update_host_i8_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      integer(8),target,dimension(:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_i8_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      integer(8),target,dimension(:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_i8_2(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      integer(8),target,dimension(:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*8_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_i8_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      integer(8),target,dimension(:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*8_8)
    end function

    function gpufort_acc_no_create_i8_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      integer(8),target,dimension(:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_i8_2(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      integer(8),target,dimension(:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_i8_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      integer(8),target,dimension(:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copyout_i8_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      integer(8),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copy_i8_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      integer(8),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    subroutine gpufort_acc_update_host_i8_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      integer(8),target,dimension(:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_i8_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      integer(8),target,dimension(:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_i8_3(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      integer(8),target,dimension(:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*8_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_i8_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      integer(8),target,dimension(:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*8_8)
    end function

    function gpufort_acc_no_create_i8_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      integer(8),target,dimension(:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_i8_3(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      integer(8),target,dimension(:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_i8_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      integer(8),target,dimension(:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copyout_i8_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      integer(8),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copy_i8_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      integer(8),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    subroutine gpufort_acc_update_host_i8_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      integer(8),target,dimension(:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_i8_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      integer(8),target,dimension(:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_i8_4(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      integer(8),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*8_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_i8_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      integer(8),target,dimension(:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*8_8)
    end function

    function gpufort_acc_no_create_i8_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      integer(8),target,dimension(:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_i8_4(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      integer(8),target,dimension(:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_i8_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      integer(8),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copyout_i8_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      integer(8),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copy_i8_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      integer(8),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    subroutine gpufort_acc_update_host_i8_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      integer(8),target,dimension(:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_i8_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      integer(8),target,dimension(:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_i8_5(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*8_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_i8_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*8_8)
    end function

    function gpufort_acc_no_create_i8_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_i8_5(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_i8_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copyout_i8_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copy_i8_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    subroutine gpufort_acc_update_host_i8_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_i8_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_i8_6(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*8_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_i8_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*8_8)
    end function

    function gpufort_acc_no_create_i8_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_i8_6(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_i8_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copyout_i8_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copy_i8_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    subroutine gpufort_acc_update_host_i8_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_i8_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_i8_7(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*8_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_i8_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*8_8)
    end function

    function gpufort_acc_no_create_i8_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_i8_7(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_i8_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copyout_i8_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copy_i8_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    subroutine gpufort_acc_update_host_i8_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_i8_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      integer(8),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

 
                                                              
    function gpufort_acc_present_r4_0(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      real(4),target,intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),4_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_r4_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      real(4),target,intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),4_8)
    end function

    function gpufort_acc_no_create_r4_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      real(4),target,intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_r4_0(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      real(4),target,intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_r4_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      real(4),target,intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),4_8,async)
    end function

    function gpufort_acc_copyout_r4_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      real(4),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),4_8,async)
    end function

    function gpufort_acc_copy_r4_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      real(4),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),4_8,async)
    end function

    subroutine gpufort_acc_update_host_r4_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      real(4),target,intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_r4_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      real(4),target,intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_r4_1(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      real(4),target,dimension(:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*4_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_r4_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      real(4),target,dimension(:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*4_8)
    end function

    function gpufort_acc_no_create_r4_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      real(4),target,dimension(:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_r4_1(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      real(4),target,dimension(:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_r4_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      real(4),target,dimension(:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copyout_r4_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      real(4),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copy_r4_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      real(4),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    subroutine gpufort_acc_update_host_r4_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      real(4),target,dimension(:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_r4_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      real(4),target,dimension(:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_r4_2(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      real(4),target,dimension(:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*4_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_r4_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      real(4),target,dimension(:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*4_8)
    end function

    function gpufort_acc_no_create_r4_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      real(4),target,dimension(:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_r4_2(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      real(4),target,dimension(:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_r4_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      real(4),target,dimension(:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copyout_r4_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      real(4),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copy_r4_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      real(4),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    subroutine gpufort_acc_update_host_r4_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      real(4),target,dimension(:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_r4_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      real(4),target,dimension(:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_r4_3(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      real(4),target,dimension(:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*4_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_r4_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      real(4),target,dimension(:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*4_8)
    end function

    function gpufort_acc_no_create_r4_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      real(4),target,dimension(:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_r4_3(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      real(4),target,dimension(:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_r4_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      real(4),target,dimension(:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copyout_r4_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      real(4),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copy_r4_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      real(4),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    subroutine gpufort_acc_update_host_r4_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      real(4),target,dimension(:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_r4_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      real(4),target,dimension(:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_r4_4(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      real(4),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*4_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_r4_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      real(4),target,dimension(:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*4_8)
    end function

    function gpufort_acc_no_create_r4_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      real(4),target,dimension(:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_r4_4(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      real(4),target,dimension(:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_r4_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      real(4),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copyout_r4_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      real(4),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copy_r4_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      real(4),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    subroutine gpufort_acc_update_host_r4_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      real(4),target,dimension(:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_r4_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      real(4),target,dimension(:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_r4_5(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      real(4),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*4_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_r4_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      real(4),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*4_8)
    end function

    function gpufort_acc_no_create_r4_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      real(4),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_r4_5(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      real(4),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_r4_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      real(4),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copyout_r4_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      real(4),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copy_r4_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      real(4),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    subroutine gpufort_acc_update_host_r4_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      real(4),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_r4_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      real(4),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_r4_6(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      real(4),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*4_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_r4_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      real(4),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*4_8)
    end function

    function gpufort_acc_no_create_r4_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      real(4),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_r4_6(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      real(4),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_r4_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      real(4),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copyout_r4_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      real(4),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copy_r4_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      real(4),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    subroutine gpufort_acc_update_host_r4_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      real(4),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_r4_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      real(4),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_r4_7(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      real(4),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*4_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_r4_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      real(4),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*4_8)
    end function

    function gpufort_acc_no_create_r4_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      real(4),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_r4_7(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      real(4),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_r4_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      real(4),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copyout_r4_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      real(4),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    function gpufort_acc_copy_r4_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      real(4),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*4_8,async)
    end function

    subroutine gpufort_acc_update_host_r4_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      real(4),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_r4_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      real(4),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

 
                                                              
    function gpufort_acc_present_r8_0(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      real(8),target,intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),8_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_r8_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      real(8),target,intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),8_8)
    end function

    function gpufort_acc_no_create_r8_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      real(8),target,intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_r8_0(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      real(8),target,intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_r8_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      real(8),target,intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),8_8,async)
    end function

    function gpufort_acc_copyout_r8_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      real(8),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),8_8,async)
    end function

    function gpufort_acc_copy_r8_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      real(8),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),8_8,async)
    end function

    subroutine gpufort_acc_update_host_r8_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      real(8),target,intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_r8_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      real(8),target,intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_r8_1(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      real(8),target,dimension(:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*8_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_r8_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      real(8),target,dimension(:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*8_8)
    end function

    function gpufort_acc_no_create_r8_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      real(8),target,dimension(:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_r8_1(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      real(8),target,dimension(:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_r8_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      real(8),target,dimension(:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copyout_r8_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      real(8),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copy_r8_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      real(8),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    subroutine gpufort_acc_update_host_r8_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      real(8),target,dimension(:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_r8_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      real(8),target,dimension(:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_r8_2(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      real(8),target,dimension(:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*8_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_r8_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      real(8),target,dimension(:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*8_8)
    end function

    function gpufort_acc_no_create_r8_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      real(8),target,dimension(:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_r8_2(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      real(8),target,dimension(:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_r8_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      real(8),target,dimension(:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copyout_r8_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      real(8),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copy_r8_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      real(8),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    subroutine gpufort_acc_update_host_r8_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      real(8),target,dimension(:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_r8_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      real(8),target,dimension(:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_r8_3(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      real(8),target,dimension(:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*8_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_r8_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      real(8),target,dimension(:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*8_8)
    end function

    function gpufort_acc_no_create_r8_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      real(8),target,dimension(:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_r8_3(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      real(8),target,dimension(:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_r8_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      real(8),target,dimension(:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copyout_r8_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      real(8),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copy_r8_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      real(8),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    subroutine gpufort_acc_update_host_r8_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      real(8),target,dimension(:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_r8_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      real(8),target,dimension(:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_r8_4(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      real(8),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*8_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_r8_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      real(8),target,dimension(:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*8_8)
    end function

    function gpufort_acc_no_create_r8_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      real(8),target,dimension(:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_r8_4(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      real(8),target,dimension(:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_r8_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      real(8),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copyout_r8_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      real(8),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copy_r8_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      real(8),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    subroutine gpufort_acc_update_host_r8_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      real(8),target,dimension(:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_r8_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      real(8),target,dimension(:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_r8_5(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      real(8),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*8_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_r8_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      real(8),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*8_8)
    end function

    function gpufort_acc_no_create_r8_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      real(8),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_r8_5(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      real(8),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_r8_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      real(8),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copyout_r8_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      real(8),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copy_r8_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      real(8),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    subroutine gpufort_acc_update_host_r8_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      real(8),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_r8_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      real(8),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_r8_6(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      real(8),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*8_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_r8_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      real(8),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*8_8)
    end function

    function gpufort_acc_no_create_r8_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      real(8),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_r8_6(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      real(8),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_r8_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      real(8),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copyout_r8_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      real(8),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copy_r8_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      real(8),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    subroutine gpufort_acc_update_host_r8_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      real(8),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_r8_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      real(8),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_r8_7(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      real(8),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*8_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_r8_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      real(8),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*8_8)
    end function

    function gpufort_acc_no_create_r8_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      real(8),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_r8_7(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      real(8),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_r8_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      real(8),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copyout_r8_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      real(8),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    function gpufort_acc_copy_r8_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      real(8),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*8_8,async)
    end function

    subroutine gpufort_acc_update_host_r8_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      real(8),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_r8_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      real(8),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

 
                                                              
    function gpufort_acc_present_c4_0(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      complex(4),target,intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),2*4_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_c4_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      complex(4),target,intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),2*4_8)
    end function

    function gpufort_acc_no_create_c4_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      complex(4),target,intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_c4_0(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      complex(4),target,intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_c4_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      complex(4),target,intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),2*4_8,async)
    end function

    function gpufort_acc_copyout_c4_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      complex(4),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),2*4_8,async)
    end function

    function gpufort_acc_copy_c4_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      complex(4),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),2*4_8,async)
    end function

    subroutine gpufort_acc_update_host_c4_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      complex(4),target,intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_c4_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      complex(4),target,intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_c4_1(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      complex(4),target,dimension(:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*2*4_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_c4_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      complex(4),target,dimension(:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*2*4_8)
    end function

    function gpufort_acc_no_create_c4_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      complex(4),target,dimension(:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_c4_1(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      complex(4),target,dimension(:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_c4_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      complex(4),target,dimension(:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*2*4_8,async)
    end function

    function gpufort_acc_copyout_c4_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      complex(4),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*2*4_8,async)
    end function

    function gpufort_acc_copy_c4_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      complex(4),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*2*4_8,async)
    end function

    subroutine gpufort_acc_update_host_c4_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      complex(4),target,dimension(:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_c4_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      complex(4),target,dimension(:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_c4_2(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      complex(4),target,dimension(:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*2*4_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_c4_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      complex(4),target,dimension(:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*2*4_8)
    end function

    function gpufort_acc_no_create_c4_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      complex(4),target,dimension(:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_c4_2(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      complex(4),target,dimension(:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_c4_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      complex(4),target,dimension(:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*2*4_8,async)
    end function

    function gpufort_acc_copyout_c4_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      complex(4),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*2*4_8,async)
    end function

    function gpufort_acc_copy_c4_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      complex(4),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*2*4_8,async)
    end function

    subroutine gpufort_acc_update_host_c4_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      complex(4),target,dimension(:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_c4_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      complex(4),target,dimension(:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_c4_3(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      complex(4),target,dimension(:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*2*4_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_c4_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      complex(4),target,dimension(:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*2*4_8)
    end function

    function gpufort_acc_no_create_c4_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      complex(4),target,dimension(:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_c4_3(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      complex(4),target,dimension(:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_c4_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      complex(4),target,dimension(:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*2*4_8,async)
    end function

    function gpufort_acc_copyout_c4_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      complex(4),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*2*4_8,async)
    end function

    function gpufort_acc_copy_c4_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      complex(4),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*2*4_8,async)
    end function

    subroutine gpufort_acc_update_host_c4_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      complex(4),target,dimension(:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_c4_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      complex(4),target,dimension(:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_c4_4(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      complex(4),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*2*4_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_c4_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      complex(4),target,dimension(:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*2*4_8)
    end function

    function gpufort_acc_no_create_c4_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      complex(4),target,dimension(:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_c4_4(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      complex(4),target,dimension(:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_c4_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      complex(4),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*2*4_8,async)
    end function

    function gpufort_acc_copyout_c4_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      complex(4),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*2*4_8,async)
    end function

    function gpufort_acc_copy_c4_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      complex(4),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*2*4_8,async)
    end function

    subroutine gpufort_acc_update_host_c4_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      complex(4),target,dimension(:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_c4_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      complex(4),target,dimension(:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_c4_5(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*2*4_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_c4_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*2*4_8)
    end function

    function gpufort_acc_no_create_c4_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_c4_5(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_c4_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*2*4_8,async)
    end function

    function gpufort_acc_copyout_c4_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*2*4_8,async)
    end function

    function gpufort_acc_copy_c4_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*2*4_8,async)
    end function

    subroutine gpufort_acc_update_host_c4_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_c4_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_c4_6(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*2*4_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_c4_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*2*4_8)
    end function

    function gpufort_acc_no_create_c4_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_c4_6(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_c4_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*2*4_8,async)
    end function

    function gpufort_acc_copyout_c4_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*2*4_8,async)
    end function

    function gpufort_acc_copy_c4_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*2*4_8,async)
    end function

    subroutine gpufort_acc_update_host_c4_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_c4_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_c4_7(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*2*4_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_c4_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*2*4_8)
    end function

    function gpufort_acc_no_create_c4_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_c4_7(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_c4_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*2*4_8,async)
    end function

    function gpufort_acc_copyout_c4_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*2*4_8,async)
    end function

    function gpufort_acc_copy_c4_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*2*4_8,async)
    end function

    subroutine gpufort_acc_update_host_c4_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_c4_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      complex(4),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

 
                                                              
    function gpufort_acc_present_c8_0(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      complex(8),target,intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),2*8_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_c8_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      complex(8),target,intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),2*8_8)
    end function

    function gpufort_acc_no_create_c8_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      complex(8),target,intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_c8_0(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      complex(8),target,intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_c8_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      complex(8),target,intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),2*8_8,async)
    end function

    function gpufort_acc_copyout_c8_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      complex(8),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),2*8_8,async)
    end function

    function gpufort_acc_copy_c8_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      complex(8),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),2*8_8,async)
    end function

    subroutine gpufort_acc_update_host_c8_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      complex(8),target,intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_c8_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      complex(8),target,intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_c8_1(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      complex(8),target,dimension(:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*2*8_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_c8_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      complex(8),target,dimension(:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*2*8_8)
    end function

    function gpufort_acc_no_create_c8_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      complex(8),target,dimension(:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_c8_1(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      complex(8),target,dimension(:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_c8_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      complex(8),target,dimension(:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*2*8_8,async)
    end function

    function gpufort_acc_copyout_c8_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      complex(8),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*2*8_8,async)
    end function

    function gpufort_acc_copy_c8_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      complex(8),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*2*8_8,async)
    end function

    subroutine gpufort_acc_update_host_c8_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      complex(8),target,dimension(:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_c8_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      complex(8),target,dimension(:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_c8_2(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      complex(8),target,dimension(:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*2*8_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_c8_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      complex(8),target,dimension(:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*2*8_8)
    end function

    function gpufort_acc_no_create_c8_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      complex(8),target,dimension(:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_c8_2(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      complex(8),target,dimension(:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_c8_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      complex(8),target,dimension(:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*2*8_8,async)
    end function

    function gpufort_acc_copyout_c8_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      complex(8),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*2*8_8,async)
    end function

    function gpufort_acc_copy_c8_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      complex(8),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*2*8_8,async)
    end function

    subroutine gpufort_acc_update_host_c8_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      complex(8),target,dimension(:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_c8_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      complex(8),target,dimension(:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_c8_3(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      complex(8),target,dimension(:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*2*8_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_c8_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      complex(8),target,dimension(:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*2*8_8)
    end function

    function gpufort_acc_no_create_c8_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      complex(8),target,dimension(:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_c8_3(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      complex(8),target,dimension(:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_c8_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      complex(8),target,dimension(:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*2*8_8,async)
    end function

    function gpufort_acc_copyout_c8_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      complex(8),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*2*8_8,async)
    end function

    function gpufort_acc_copy_c8_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      complex(8),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*2*8_8,async)
    end function

    subroutine gpufort_acc_update_host_c8_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      complex(8),target,dimension(:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_c8_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      complex(8),target,dimension(:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_c8_4(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      complex(8),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*2*8_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_c8_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      complex(8),target,dimension(:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*2*8_8)
    end function

    function gpufort_acc_no_create_c8_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      complex(8),target,dimension(:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_c8_4(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      complex(8),target,dimension(:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_c8_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      complex(8),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*2*8_8,async)
    end function

    function gpufort_acc_copyout_c8_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      complex(8),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*2*8_8,async)
    end function

    function gpufort_acc_copy_c8_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      complex(8),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*2*8_8,async)
    end function

    subroutine gpufort_acc_update_host_c8_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      complex(8),target,dimension(:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_c8_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      complex(8),target,dimension(:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_c8_5(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*2*8_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_c8_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*2*8_8)
    end function

    function gpufort_acc_no_create_c8_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_c8_5(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_c8_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*2*8_8,async)
    end function

    function gpufort_acc_copyout_c8_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*2*8_8,async)
    end function

    function gpufort_acc_copy_c8_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*2*8_8,async)
    end function

    subroutine gpufort_acc_update_host_c8_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_c8_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_c8_6(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*2*8_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_c8_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*2*8_8)
    end function

    function gpufort_acc_no_create_c8_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_c8_6(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_c8_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*2*8_8,async)
    end function

    function gpufort_acc_copyout_c8_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*2*8_8,async)
    end function

    function gpufort_acc_copy_c8_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*2*8_8,async)
    end function

    subroutine gpufort_acc_update_host_c8_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_c8_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                              
    function gpufort_acc_present_c8_7(hostptr,async,copy,copyin,create) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_present_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      logical,intent(in),optional :: copy
      logical,intent(in),optional :: copyin
      logical,intent(in),optional :: create
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_present_b(c_loc(hostptr),size(hostptr)*2*8_8,async,copy,copyin,create)
    end function

    function gpufort_acc_create_c8_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_create_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_create_b(c_loc(hostptr),size(hostptr)*2*8_8)
    end function

    function gpufort_acc_no_create_c8_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_no_create_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufort_acc_delete_c8_7(hostptr,finalize)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_delete_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: finalize
      !
      call gpufort_acc_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufort_acc_copyin_c8_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyin_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyin_b(c_loc(hostptr),size(hostptr)*2*8_8,async)
    end function

    function gpufort_acc_copyout_c8_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copyout_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copyout_b(c_loc(hostptr),size(hostptr)*2*8_8,async)
    end function

    function gpufort_acc_copy_c8_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_copy_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufort_acc_copy_b(c_loc(hostptr),size(hostptr)*2*8_8,async)
    end function

    subroutine gpufort_acc_update_host_c8_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_host_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

    subroutine gpufort_acc_update_device_c8_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufort_acc_runtime_base, only: gpufort_acc_update_device_b
      implicit none
      complex(8),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufort_acc_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

 
 

end module