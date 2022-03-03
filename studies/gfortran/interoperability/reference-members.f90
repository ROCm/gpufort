! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
program main
  use iso_c_binding
  implicit none

  type, bind(c) :: B
    real(c_double) :: d1
    integer(c_int) :: i2
  end type

  type, bind(c) :: A
    integer(c_int) :: i1
    type(c_ptr) :: i2
    type(c_ptr) :: b3
  end type
  !
  type(B),target :: t_B = B(3,4)
  type(A) :: t_A
  !
  integer,target :: i
  !
  i = 2
  t_A%i1 = 1
  t_A%i2 = c_loc(i)
  t_A%b3 = c_loc(t_B)
  ! call C routine
  call print_struct(t_A)

end program
