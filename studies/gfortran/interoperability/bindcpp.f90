! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
module mylib
interface ! bind(c) procedures can be contained in Fortran program
  ! implicitly defined
  !subroutine mysubroutine2(a,b) bind(c,name="mysubroutine2")
  !  implicit none
  !  integer,intent(in)    :: a
  !  integer,intent(inout) :: b
  !end subroutine
  
  function myfunction2(a,b) result(res) bind(c,name="myfunction2") 
    implicit none
    integer,intent(in)    :: a
    integer,intent(inout) :: b
    !
    integer :: res
  end function
end interface 

contains ! bind(c) procedures cannot be contained in Fortran program
         ! so must move them in their own module
  subroutine mysubroutine1(a,b) bind(c,name="mysubroutine1")
    implicit none
    integer,intent(in)    :: a
    integer,intent(inout) :: b
    b = a + b
  end subroutine
  
  function myfunction1(a,b) result(res) bind(c,name="myfunction1")
    implicit none
    integer,intent(in)    :: a
    integer,intent(inout) :: b
    !
    integer :: res
    b = a + 2*b
    res = -5
  end function
end module

program myprogram
  use mylib
  implicit none
  integer :: a,b
  integer :: res
  
  a = 5
  b = 2
  
  call mysubroutine2(a,b)
  
  print *, "Fortran code: value of 'b' [post]: ", b ! should be 7
  
  b = 2
  res = myfunction2(a,b)
  
  print *, "Fortran code: value of 'b' [post]: ", b ! should be 9
  print *, "Fortran code: value of 'res' [post]: ", res ! should be -5
end program