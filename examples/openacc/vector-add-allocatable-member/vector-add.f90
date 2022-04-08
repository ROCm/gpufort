! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
program main
  ! begin of program
      
  implicit none
  integer, parameter :: N = 1000
  integer :: i
  integer(4) :: y(N), y_exact(N)
  type struct_t
    integer(4) :: coeff
    integer(4),allocatable :: x(:)
  end type
  type(struct_t) :: struct

  struct%coeff = 1
  allocate(struct%x(N))

  do i = 1, N
    y_exact(i) = 3
  end do

  !$acc data copy(struct%x(1:N),y(1:N))
 
  !$acc parallel loop present(struct%x(1:N),y(1:N))
  do i = 1, N
    struct%x(i) = 1
    y(i) = 2
  end do
  
  !$acc parallel loop
  do i = 1, N
    y(i) = struct%x(i) + struct%coeff*y(i)
  end do
  !$acc end data
  
  do i = 1, N
    if ( y_exact(i) .ne.&
            y(i) ) ERROR STOP "GPU and CPU result do not match"
  end do

  deallocate(struct%x)

  print *, "PASSED"

end program
