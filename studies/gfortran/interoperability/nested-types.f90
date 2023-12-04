! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
program main
  use iso_c_binding
  implicit none
  integer :: i

  type, bind(c) :: B
    integer(4) :: mb
  end type
  
  type, bind(c)    :: A
    type(B)        :: tb
    type(B)        :: tb_static_array(5)
    ! generate per array
    type(c_ptr)    :: tb_dyn_array = c_null_ptr
    integer(c_int) :: tb_dyn_array_n1
    integer(c_int) :: tb_dyn_array_lb1
  end type
  
  interface 
     subroutine read_nested_struct(ta) bind(c,name="read_nested_struct")
       import A 
       type(A) :: ta
     end subroutine
  end interface
  
  type(A),target  :: ta
  !
  type(c_ptr)     :: cptr = c_null_ptr
  type(A),pointer :: fptr => null()
  !
  type(B),allocatable,dimension(:) :: tb_dyn_array
  
  ta%tb%mb = 251
  
  ! call C routine
  call read_nested_struct(ta)
  
  ! cast c_ptr to Fortran derived type ptr
  cptr = c_loc(ta)
  call c_f_pointer(cptr,fptr)
  print *, fptr%tb%mb
  
  ! cast
  allocate(tb_dyn_array(-5:5))

  do i=-5,5
    tb_dyn_array(i)%mb = i
  end do
  !print *, lbound(tb_dyn_array)

  call init_a_tb_dyn_array(ta,tb_dyn_array,lbound(tb_dyn_array,1))
  ! bounds must be passed too. They might get "lost" in subroutine call and reset to 1 (strange)
  
  call read_nested_struct(ta)
contains
  
  ! generate
  subroutine init_a_tb_dyn_array(ta,initial,initial_lb1)
     type(A)                     :: ta
     type(B),target,dimension(:) :: initial 
     integer                     :: initial_lb1
     !
     !ta%tb_dyn_array_lb1 = lbound(initial,1)
     !print *, lbound(initial,1)
     ta%tb_dyn_array_n1  = size(initial,1)
     ta%tb_dyn_array_lb1 = initial_lb1
     ta%tb_dyn_array     = c_loc(initial)
     !ta%tb_dyn_array     = c_loc(initial(ta%tb_dyn_array_lb1))
  end subroutine

end program