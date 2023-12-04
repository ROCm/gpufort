! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
program main
  implicit none
  integer, parameter :: N = 20
  integer :: i
  integer(4) :: x(N), y(N), res

  !$acc data copy(x(1:N),y(1:N))
  
  !$acc parallel loop
  do i = 1, N
    x(i) = 1
    y(i) = 2
  end do

  res = 0
  !$acc parallel loop reduction(+:res)
  do i = 1, N
    res = res + x(i) * y(i)
  end do
  
  !$acc end data

  if ( res .ne. N*2 ) ERROR STOP "FAILED"
  PRINT *, "PASSED"

end program