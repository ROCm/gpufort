! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
program main
  ! begin of program
      
  implicit none
  integer, parameter :: N = 1000
  integer :: i,j
  integer(4) :: x(N), y(N), y_exact(N)

  integer :: add, mult, a, b
  add(a,b)=a+b ! statement functions
  mult(a,b)=a*b ! statement functions

  do i = 1, N
    y_exact(i) = 3
  end do

  !$acc data copy(x(1:N),y(1:N))

  ! just an unnecessarily complicated way to fill 1D arrays
  !$acc parallel loop present(x,y) collapse(2)
  do j = 4, -4, -8
    do i = 1, N/2
      x( ((j-4)/-8)*N/2+i ) = 1
      y( ((j-4)/-8)*N/2+i ) = 2
    end do
  end do
  
  !$acc parallel loop
  do i = 1, N
    y(i) = add(x(i),mult(1,y(i)))
  end do
  !$acc end data
  
  do i = 1, N
    if ( y_exact(i) .ne.&
            y(i) ) ERROR STOP "GPU and CPU result do not match"
  end do

  print *, "PASSED"

end program
