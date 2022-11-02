! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
! This file solely exists to extract information about intrinsics via GPUFORT's indexer tool.
! It will not compile with any compiler as non-standard Fortran expressions are used.
attributes(intrinsic,transformational,std_f2018)&
function failed_images(team,kind)
  type(c_ptr), optional :: team
  integer, optional :: kind
  !
  integer(real64) :: failed_images
end function

attributes(intrinsic,transformational,std_f2018)&
function get_team(level)
  integer, optional :: level
  !
  integer :: get_team
end function

attributes(intrinsic,std_f2018)&
elemental function image_status(image,team)
  integer :: image
  type(c_ptr), optional :: team
  !
  integer :: image_status
end function

attributes(intrinsic,inquiry,std_f2018)&
function rank(a)
  real :: a
  !
  integer :: rank
end function

attributes(intrinsic,transformational,std_f2018)&
function stopped_images(team,kind)
  type(c_ptr), optional :: team
  integer, optional :: kind
  !
  integer(real64) :: stopped_images
end function

attributes(intrinsic,transformational,std_f2018)&
function team_number(team)
  type(team_type), optional :: team
  !
  integer :: team_number
end function

attributes(intrinsic,atomic,std_f2018)&
subroutine atomic_cas(atom,old,compare,new,stat)
  integer, intent(inout) :: atom
  integer, intent(out) :: old
  integer, intent(in) :: compare
  integer, intent(in) :: new
  integer, intent(out), optional :: stat
end subroutine

attributes(intrinsic,atomic,std_f2018)&
subroutine atomic_add(atom,value,stat)
  integer, intent(out) :: atom
  integer, intent(in) :: value
  integer, intent(out), optional :: stat
end subroutine

attributes(intrinsic,atomic,std_f2018)&
subroutine atomic_and(atom,value,stat)
  integer, intent(out) :: atom
  integer, intent(in) :: value
  integer, intent(out), optional :: stat
end subroutine

attributes(intrinsic,atomic,std_f2018)&
subroutine atomic_or(atom,value,stat)
  integer, intent(out) :: atom
  integer, intent(in) :: value
  integer, intent(out), optional :: stat
end subroutine

attributes(intrinsic,atomic,std_f2018)&
subroutine atomic_xor(atom,value,stat)
  integer, intent(out) :: atom
  integer, intent(in) :: value
  integer, intent(out), optional :: stat
end subroutine

attributes(intrinsic,atomic,std_f2018)&
subroutine atomic_fetch_add(atom,value,old,stat)
  integer, intent(out) :: atom
  integer, intent(in) :: value
  integer, intent(out) :: old
  integer, intent(out), optional :: stat
end subroutine

attributes(intrinsic,atomic,std_f2018)&
subroutine atomic_fetch_and(atom,value,old,stat)
  integer, intent(out) :: atom
  integer, intent(in) :: value
  integer, intent(out) :: old
  integer, intent(out), optional :: stat
end subroutine

attributes(intrinsic,atomic,std_f2018)&
subroutine atomic_fetch_or(atom,value,old,stat)
  integer, intent(out) :: atom
  integer, intent(in) :: value
  integer, intent(out) :: old
  integer, intent(out), optional :: stat
end subroutine

attributes(intrinsic,atomic,std_f2018)&
subroutine atomic_fetch_xor(atom,value,old,stat)
  integer, intent(out) :: atom
  integer, intent(in) :: value
  integer, intent(out) :: old
  integer, intent(out), optional :: stat
end subroutine

attributes(intrinsic,atomic,std_f2018)&
subroutine event_query(event,count,stat)
  integer, intent(in) :: event
  integer, intent(in), optional :: count
  integer, intent(out), optional :: stat
end subroutine

attributes(intrinsic,impure,std_f2018)&
subroutine random_init(repeatable,image_distinct)
  logical, intent(in) :: repeatable
  logical, intent(in) :: image_distinct
end subroutine

attributes(intrinsic,impure,std_f2018)&
subroutine co_broadcast(a,source_image,stat,errmsg)
  real, intent(inout) :: a
  integer, intent(in) :: source_image
  integer, intent(out), optional :: stat
  character, intent(inout), optional :: errmsg
end subroutine

attributes(intrinsic,impure,std_f2018)&
subroutine co_max(a,result_image,stat,errmsg)
  real, intent(inout) :: a
  integer, intent(in), optional :: result_image
  integer, intent(out), optional :: stat
  character, intent(inout), optional :: errmsg
end subroutine

attributes(intrinsic,impure,std_f2018)&
subroutine co_min(a,result_image,stat,errmsg)
  real, intent(inout) :: a
  integer, intent(in), optional :: result_image
  integer, intent(out), optional :: stat
  character, intent(inout), optional :: errmsg
end subroutine

attributes(intrinsic,impure,std_f2018)&
subroutine co_sum(a,result_image,stat,errmsg)
  real, intent(inout) :: a
  integer, intent(in), optional :: result_image
  integer, intent(out), optional :: stat
  character, intent(inout), optional :: errmsg
end subroutine

attributes(intrinsic,impure,std_f2018)&
subroutine co_reduce(a,operation,result_image,stat,errmsg)
  real, intent(inout) :: a
  integer, intent(in) :: operation
  integer, intent(in), optional :: result_image
  integer, intent(out), optional :: stat
  character, intent(inout), optional :: errmsg
end subroutine

