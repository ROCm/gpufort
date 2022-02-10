! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
program main

integer, parameter :: param = 2

integer g,h,i
integer :: d,e,f
!integer a=1,b,c=2 ! illegal
integer :: a=1,b(5),c=2
integer j,k(6),l(8,8)

type test
  integer a
  integer(4) b
  integer(4*2) ba
  integer(4*param-2+2) bb 
  integer*4 c
  integer(kind(4))   d
  integer(kind(4*2)) da
  integer(kind(4*param-2+2)) db
  integer(kind=4)   e
  integer(kind=4*2) ea
  integer(kind=4*param-2+2) eb
  integer :: f
  integer(4) :: g
  integer*4 :: h
  !integer*(4+4) :: ha ! illegal
  integer(kind(4)) :: i
  integer(kind=4) :: j
  !integer, pointer k ! illegal
  integer, pointer :: l
  character(4) ma
  character(4*2) mb
  character*4 :: na
  character*(4*2) nb
  character(len=4) oa
  character(len=4*2+param-2) ob
  character :: q(4)
end type

type(test) t1
type(test) :: t2
! type*test C             ! illegal
! type*test :: D          ! illegal
! type(kind=test) J       ! illegal
! type(kind=test) :: K    ! illegal

end program