! SPDX-License-Identifier: MIT                                                
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
! 
! 
!                             Online Fortran Compiler.
!                 Code, Compile, Run and Debug Fortran program online.
! Write your code in this editor and press "Run" button to execute it.
! 
! 


Program Hello
 use iso_c_binding
 IMPLICIT none
 
 integer,ALLOCATABLE :: inp0(:)
 integer,ALLOCATABLE,target :: inp1(:)
 integer,POINTER :: inp2(:) => null()

 allocate(inp1(10))

 inp2 => inp1

 call test1(inp0)
 call test1(inp1)
 call test1(inp2)

contains

subroutine test1(inp)
  use iso_c_binding
  IMPLICIT NONE
  integer, target :: inp(:)
  !
  type(c_ptr) :: cptr
  
  cptr = c_loc(inp)
  
  print *, c_associated(cptr)
end subroutine

subroutine test2(inp)
  use iso_c_binding
  IMPLICIT NONE
  integer, pointer :: inp(:)
  !
  type(c_ptr) :: cptr
  
  cptr = c_loc(inp)
  
  print *, c_associated(cptr)
end subroutine

End Program Hello