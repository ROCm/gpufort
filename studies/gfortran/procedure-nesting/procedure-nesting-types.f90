! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
! in program units such as 'program', 'subroutine' there must be only one `contains` statement
! and it must be in a top-level module,program, or procedure.
!
! two levels of nesting (via contains) are only supported within a
! module as a module is not a program unit but instead a collection of program units
! (and interfaces, global variables/parameters, derived type definitions.)
subroutine level_1_proc()
  type :: type1_t
    integer :: a
  end type
  type(type1_t) :: type1
  !
  type1%a = 1
  call level_2_proc_1 ! call nested subroutine
contains 
  subroutine level_2_proc_1()
    use iso_c_binding
    type :: type21_t
      integer :: a
      real,pointer,dimension(:) :: b
    end type
    type(type21_t),target :: type21 
    !
    type21%a = 21
    allocate(type21%b(-1:10))
    type21%b(:) = 2.0
    print *, ">level_2_proc_1"
    print *, "type21%a:", type21%a
    call level_2_proc_2_typecast(c_loc(type21)) ! call other nested subroutine defined below; order is not important
  end subroutine
  subroutine level_2_proc_2_typecast(arg)
    use iso_c_binding
    type :: type21_t ! redeclaration of (compare to level_2_proc_1)
      integer :: a
      real,pointer,dimension(:) :: b
    end type
    type(c_ptr),intent(in) :: arg
    type(type21_t),pointer :: type21
    type(type1_t)  :: type1  ! hides variable in parent 
    ! 
    type1%a = -1
    call c_f_pointer(arg,type21)
    print *, ">level_2_proc_2:"
    print *, "type1%a:", type1%a
    print *, "type21%a:", type21%a
    print *, "lbound(type21%b,1):", lbound(type21%b,1)
    print *, "ubound(type21%b,1):", ubound(type21%b,1)
    print *, "size(type21%b,1):", size(type21%b,1)
    print *, "type21%b(-1):", type21%b(-1)
  end subroutine
end subroutine

program hello
  call level_1_proc()
end program hello