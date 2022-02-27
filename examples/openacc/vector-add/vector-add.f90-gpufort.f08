#ifdef _GPUFORT
! This file was generated by GPUFORT
#endif
! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
program main
  ! begin of program
      
#ifdef _GPUFORT
  use gpufort_acc_runtime
  use iso_c_binding
#endif
  implicit none
  integer, parameter :: N = 1000
  integer :: i
  integer(4) :: x(N), y(N), y_exact(N)
#ifdef _GPUFORT
  type(c_ptr) :: dev_y
  type(c_ptr) :: dev_x
#endif

  do i = 1, N
    y_exact(i) = 3
  end do

#ifdef _GPUFORT
  call gpufort_acc_enter_region()
  dev_x = gpufort_acc_copy(x(1:N))
  dev_y = gpufort_acc_copy(y(1:N))
 
  call gpufort_acc_enter_region()
  dev_x = gpufort_acc_present(x)
  dev_y = gpufort_acc_present(y)
  ! extracted to HIP C++ file
  call launch_main_17_auto(0,c_null_ptr,)
  call gpufort_acc_wait()
  call gpufort_acc_exit_region()
  
  call gpufort_acc_enter_region()
  dev_y = gpufort_acc_present(y,or=gpufort_acc_event_copy)
  dev_x = gpufort_acc_present(x,or=gpufort_acc_event_copy)
  ! extracted to HIP C++ file
  call launch_main_23_auto(0,c_null_ptr,)
  call gpufort_acc_wait()
  call gpufort_acc_exit_region()
  call gpufort_acc_exit_region()
#else
  !$acc data copy(x(1:N),y(1:N))
 
  !$acc parallel loop present(x,y)
  do i = 1, N
    x(i) = 1
    y(i) = 2
  end do
  
  !$acc parallel loop
  do i = 1, N
    y(i) = x(i) + y(i)
  end do
  !$acc end data
#endif
  
  do i = 1, N
    if ( y_exact(i) .ne.&
            y(i) ) ERROR STOP "GPU and CPU result do not match"
  end do

  print *, "PASSED"

end program
