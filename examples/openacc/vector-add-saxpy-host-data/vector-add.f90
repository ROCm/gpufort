! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
program main
  ! begin of program
  use hipfort_check
  use hipfort_hipblas  

  implicit none
  integer, parameter :: N = 1000
  integer :: i
  real(c_float) :: x(N), y(N), y_exact(N) 

  do i = 1, N
    y_exact(i) = 3
  end do

  !$acc data copy(x(1:N),y(1:N))
 
  !$acc parallel loop present(x,y)
  do i = 1, N
    x(i) = 1
    y(i) = 2
  end do
  
  !$acc host_data use_device(x,y)
  call hipCheck(hipblasSaxpy(handle,N,&
                             1.0,x,1,y,1))
  !$acc end host_data
  
  do i = 1, N
    if ( y_exact(i) .ne.&
            y(i) ) ERROR STOP "GPU and CPU result do not match"
  end do

  print *, "PASSED"

end program
