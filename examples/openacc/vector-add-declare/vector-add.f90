! SPDX-License-Identifier: MIT
! Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
program main
  ! begin of program
      
  implicit none
  integer, parameter :: N = 1000
  integer :: i
  integer(4) :: x(N), y(N), y_exact(N)
  !$acc declare create(x,y)

  y_exact = 3

  !$acc parallel loop
  do i = 1, N
    x(i) = 1
    y(i) = 2
  end do
  
  !$acc parallel loop
  do i = 1, N
    y(i) = x(i) + y(i)
  end do
 
  !$acc update host(y)
 
  do i = 1, N
    if ( y_exact(i) .ne.&
            y(i) ) ERROR STOP "GPU and CPU result do not match"
  end do

  print *, "PASSED"

end program