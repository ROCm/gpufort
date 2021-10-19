! SPDX-License-Identifier: MIT                                                
! Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
program test_acc
  use iso_c_binding
  use gpufort_acc_runtime
  !
  real(8),allocatable,target :: a(:,:), b(:,:)
  type(c_ptr) :: a_d, b_d
  integer :: i
  !
  allocate(a(10,10))
  allocate(b(10,9))

  print *, "initializing"
  !$acc init
  call gpufort_acc_init()
 
  print *, "entering region" 
  call gpufort_acc_enter_region()
  a_d = gpufort_acc_copyin(a(:,:))
  call gpufort_acc_exit_region()
  print *, "exiting region" 
  
  print *, "entering region" 
  call gpufort_acc_enter_region()
  b_d = gpufort_acc_copyin(b(:,:))
  call gpufort_acc_exit_region()
  print *, "exiting region" 

  call gpufort_acc_runtime_print_summary() 
  
  print *, "shutting down"
  !$acc shutdown
  call gpufort_acc_shutdown()

end program 
