! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
program main
  ! begin of program
      
  implicit none
  integer, parameter :: N = 1000
  integer :: i,j
  integer(4) :: x(N,N), y(N,N), y_exact(N)

  y_exact = 3

  x = 1
  y = 2
  
  call vector_add_gpu(x(1:,3),y(1:,3),.FALSE.)
  call vector_add_gpu(x(1:,3),y(1:,3),.TRUE.) ! return prematurely; do not perform addition
 
  do i = 1, N
    if ( y_exact(i) .ne.&
            y(i,3) ) ERROR STOP "GPU and CPU result do not match"
  end do

  print *, "PASSED"

contains

subroutine vector_add_gpu(x,y,return_prematurely)
  integer(4) :: x(:), y(:)
  logical    :: return_prematurely
  !$acc declare copyin(x,y)
  
  ! extra return statement
  if ( return_prematurely ) return
  
  !$acc parallel loop
  do i = 1, N
    y(i) = x(i) + y(i)
  end do
  
  !$acc update host(y)
end subroutine

end program
