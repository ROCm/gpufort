! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
! This file solely exists to extract information about intrinsics via GPUFORT's indexer tool.
! It will not compile with any compiler as non-standard Fortran expressions are used.
attributes(intrinsic,inquiry,std_f2003)&
function command_argument_count()
  !
  integer :: command_argument_count
end function

attributes(intrinsic,inquiry,std_f2003)&
function extends_type_of(a,mold)
  type(*), dimension(..) :: a
  type(*), dimension(..) :: mold
  !
  logical :: extends_type_of
end function

attributes(intrinsic,std_f2003)&
elemental function is_iostat_end(i)
  integer :: i
  !
  logical :: is_iostat_end
end function

attributes(intrinsic,std_f2003)&
elemental function is_iostat_eor(i)
  integer :: i
  !
  logical :: is_iostat_eor
end function

attributes(intrinsic,inquiry,std_f2003)&
function new_line(a)
  character :: a
  !
  character :: new_line
end function

attributes(intrinsic,inquiry,std_f2003)&
function same_type_as(a,b)
  type(*), dimension(..) :: a
  type(*), dimension(..) :: b
  !
  logical :: same_type_as
end function

attributes(intrinsic,transformational,std_f2003)&
function selected_char_kind(name)
  character :: name
  !
  integer :: selected_char_kind
end function

attributes(intrinsic,inquiry,std_f2003)&
function c_associated(c_ptr_1,c_ptr_2)
  type(c_ptr) :: c_ptr_1
  type(c_ptr), optional :: c_ptr_2
  !
  logical :: c_associated
end function

attributes(intrinsic,inquiry,std_f2003)&
function c_loc(x)
  type(*), dimension(..) :: x
  !
  type(c_ptr) :: c_loc
end function

attributes(intrinsic,inquiry,std_f2003)&
function c_funloc(x)
  type(*), dimension(..) :: x
  !
  type(c_ptr) :: c_funloc
end function

attributes(intrinsic,impure,std_f2003)&
subroutine get_command(command,length,status)
  character, intent(out), optional :: command
  integer, intent(out), optional :: length
  integer, intent(out), optional :: status
end subroutine

attributes(intrinsic,impure,std_f2003)&
subroutine get_command_argument(number,value,length,status)
  integer, intent(in) :: number
  character, intent(out), optional :: value
  integer, intent(out), optional :: length
  integer, intent(out), optional :: status
end subroutine

attributes(intrinsic,impure,std_f2003)&
subroutine get_environment_variable(name,value,length,status,trim_name)
  character, intent(in) :: name
  character, intent(out), optional :: value
  integer, intent(out), optional :: length
  integer, intent(out), optional :: status
  logical, intent(in), optional :: trim_name
end subroutine

attributes(intrinsic,std_f2003)&
pure subroutine move_alloc(from,to)
  type(*), dimension(..), intent(inout) :: from
  type(*), dimension(..), intent(out) :: to
end subroutine

attributes(intrinsic,impure,std_f2003)&
subroutine c_f_pointer(cptr,fptr,shape)
  type(c_ptr), intent(in) :: cptr
  type(*), dimension(..), intent(out) :: fptr
  integer, intent(in), optional :: shape
end subroutine

attributes(intrinsic,impure,std_f2003)&
subroutine c_f_procpointer(cptr,fptr)
  type(c_ptr), intent(in) :: cptr
  type(*), dimension(..), intent(out) :: fptr
end subroutine

