! SPDX-License-Identifier: MIT                                                
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
program test_acc
  use iso_c_binding
  use gpufortrt
  !
  real(8),allocatable,target :: a(:,:), b(:,:)
  type(c_ptr) :: a_d, b_d
  integer :: i
  !
  allocate(a(10,10))
  allocate(b(10,9))

  print *, "initializing"
  !$acc init
  call gpufortrt_init()
 
  print *, "entering region" 
  call gpufortrt_enter_region()
  a_d = gpufortrt_copyin(a(:,:))
  call gpufortrt_exit_region()
  print *, "exiting region" 
  
  print *, "entering region" 
  call gpufortrt_enter_region()
  b_d = gpufortrt_copyin(b(:,:))
  call gpufortrt_exit_region()
  print *, "exiting region" 

  call gpufortrt_print_summary() 
  
  print *, "shutting down"
  !$acc shutdown
  call gpufortrt_shutdown()

end program 