! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
program main
  ! begin of program
      
  implicit none
  integer, parameter :: N = 1000
  integer :: i,j
  integer(4) :: x(N), y(N,5), y_exact(N)

  integer :: add, mult, a, b
  add(a,b)=a+b ! statement functions
  mult(a,b)=a*b ! statement functions

  do i = 1, N
    y_exact(i) = 3
  end do

  !$acc data copy(x(1:N),y(1:N,1))

  ! just an unnecessarily complicated way to fill 1D arrays
  !!$acc parallel loop present(x,y(1:N,1)) & 
  !!$acc collapse(2)
  !$acc parallel loop present(x,y(1:N,1)) & ! inline comment 
  !$acc collapse(2) ! continuation of the above directive
  do j = 4, -4, -8
    do 50 i = 1, N/2
      x( ((j-4)/-8)*N/2+i ) = 1
      y( ((j-4)/-8)*N/2+i, 1 ) = 2
 50 continue
  end do
  
  !$acc parallel loop present(x,y(1:N,1))
  do i = 1, N
    y(i,1) = add(x(i),mult(1,y(i,1)))
  end do
  !$acc end data
  
  do i = 1, N
    if ( y_exact(i) .ne.&
            y(i,1) ) ERROR STOP "GPU and CPU result do not match"
  end do

  print *, "PASSED"

end program
