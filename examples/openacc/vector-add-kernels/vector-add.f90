! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
program main
  ! begin of program
      
  implicit none
  integer, parameter :: N = 1000
  integer :: i
  integer(4) :: x(N), y(N), y_exact(N)

  do i = 1, N
    y_exact(i) = 3
  end do

  !$acc kernels copy(x,y)
  x = 1

  y(1:N) = x(1:n) + x

  do i = 1, n
    y(i) = x(i) + y(i) + 1
  end do

  !$acc loop
  do i = 1, n
    y(i) = y(i) - 1
  end do
  !$acc end kernels
  
  do i = 1, N
    if ( y_exact(i) .ne.&
            y(i) ) ERROR STOP "GPU and CPU result do not match"
  end do

  print *, "PASSED"

end program
