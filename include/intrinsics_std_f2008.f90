! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
! This file solely exists to extract information about intrinsics via GPUFORT's indexer tool.
! It will not compile with any compiler as non-standard Fortran expressions are used.
attributes(intrinsic,std_f2008,actual)&
elemental function acosh(x)
  real :: x
  !
  real :: acosh
end function

attributes(intrinsic,std_f2008,actual)&
elemental function asinh(x)
  real :: x
  !
  real :: asinh
end function

attributes(intrinsic,std_f2008,actual)&
elemental function atan(y,x)
  real :: y
  real :: x
  !
  real :: atan
end function

attributes(intrinsic,std_f2008,actual)&
elemental function atanh(x)
  real :: x
  !
  real :: atanh
end function

attributes(intrinsic,std_f2008)&
elemental function bessel_j0(x)
  real :: x
  !
  real :: bessel_j0
end function

attributes(intrinsic,std_f2008)&
elemental function bessel_j1(x)
  real :: x
  !
  real :: bessel_j1
end function

attributes(intrinsic,std_f2008)&
elemental function bessel_jn(n,x)
  integer :: n
  real :: x
  !
  real :: bessel_jn
end function

attributes(intrinsic,transformational,std_f2008)&
function bessel_jn(n1,n2,x)
  integer :: n1
  integer :: n2
  real :: x
  !
  real :: bessel_jn
end function

attributes(intrinsic,std_f2008)&
elemental function bessel_y0(x)
  real :: x
  !
  real :: bessel_y0
end function

attributes(intrinsic,std_f2008)&
elemental function bessel_y1(x)
  real :: x
  !
  real :: bessel_y1
end function

attributes(intrinsic,std_f2008)&
elemental function bessel_yn(n,x)
  integer :: n
  real :: x
  !
  real :: bessel_yn
end function

attributes(intrinsic,transformational,std_f2008)&
function bessel_yn(n1,n2,x)
  integer :: n1
  integer :: n2
  real :: x
  !
  real :: bessel_yn
end function

attributes(intrinsic,std_f2008)&
elemental function bge(i,j)
  integer :: i
  integer :: j
  !
  logical :: bge
end function

attributes(intrinsic,std_f2008)&
elemental function bgt(i,j)
  integer :: i
  integer :: j
  !
  logical :: bgt
end function

attributes(intrinsic,std_f2008)&
elemental function ble(i,j)
  integer :: i
  integer :: j
  !
  logical :: ble
end function

attributes(intrinsic,std_f2008)&
elemental function blt(i,j)
  integer :: i
  integer :: j
  !
  logical :: blt
end function

attributes(intrinsic,std_f2008)&
elemental function dshiftl(i,j,shift)
  integer :: i
  integer :: j
  integer :: shift
  !
  integer :: dshiftl
end function

attributes(intrinsic,std_f2008)&
elemental function dshiftr(i,j,shift)
  integer :: i
  integer :: j
  integer :: shift
  !
  integer :: dshiftr
end function

attributes(intrinsic,std_f2008)&
elemental function erf(x)
  real :: x
  !
  real :: erf
end function

attributes(intrinsic,std_f2008)&
elemental function erfc(x)
  real :: x
  !
  real :: erfc
end function

attributes(intrinsic,std_f2008)&
elemental function erfc_scaled(x)
  real :: x
  !
  real :: erfc_scaled
end function

attributes(intrinsic,std_f2008)&
elemental function gamma(x)
  real :: x
  !
  real :: gamma
end function

attributes(intrinsic,std_f2008)&
elemental function hypot(x,y)
  real :: x
  real :: y
  !
  real :: hypot
end function

attributes(intrinsic,transformational,std_f2008)&
function iall(array,dim,mask)
  real :: array
  integer, optional :: dim
  logical, optional :: mask
  !
  real :: iall
end function

attributes(intrinsic,transformational,std_f2008)&
function iany(array,dim,mask)
  real :: array
  integer, optional :: dim
  logical, optional :: mask
  !
  real :: iany
end function

attributes(intrinsic,inquiry,std_f2008)&
function image_index(coarray,sub)
  real :: coarray
  integer :: sub
  !
  integer :: image_index
end function

attributes(intrinsic,transformational,std_f2008)&
function iparity(array,dim,mask)
  real :: array
  integer, optional :: dim
  logical, optional :: mask
  !
  real :: iparity
end function

attributes(intrinsic,inquiry,std_f2008)&
function is_contiguous(array)
  real :: array
  !
  logical :: is_contiguous
end function

attributes(intrinsic,inquiry,std_f2008)&
function lcobound(coarray,dim,kind)
  real :: coarray
  integer, optional :: dim
  integer, optional :: kind
  !
  integer :: lcobound
end function

attributes(intrinsic,std_f2008)&
elemental function leadz(i)
  integer :: i
  !
  integer :: leadz
end function

attributes(intrinsic,std_f2008)&
elemental function log_gamma(x)
  real :: x
  !
  real :: log_gamma
end function

attributes(intrinsic,std_f2008)&
elemental function maskl(i,kind)
  integer :: i
  integer, optional :: kind
  !
  integer :: maskl
end function

attributes(intrinsic,std_f2008)&
elemental function maskr(i,kind)
  integer :: i
  integer, optional :: kind
  !
  integer :: maskr
end function

attributes(intrinsic,transformational,std_f2008)&
function findloc(array,value,dim,mask,kind,back)
  real :: array
  real :: value
  integer, optional :: dim
  logical, optional :: mask
  integer, optional :: kind
  logical, optional :: back
  !
  integer :: findloc
end function

attributes(intrinsic,std_f2008)&
elemental function merge_bits(i,j,mask)
  integer :: i
  integer :: j
  integer :: mask
  !
  integer :: merge_bits
end function

attributes(intrinsic,transformational,std_f2008)&
function norm2(x,dim)
  real :: x
  integer, optional :: dim
  !
  real :: norm2
end function

attributes(intrinsic,transformational,std_f2008)&
function num_images(distance,failed)
  integer, optional :: distance
  logical, optional :: failed
  !
  integer :: num_images
end function

attributes(intrinsic,transformational,std_f2008)&
function parity(mask,dim)
  logical :: mask
  integer, optional :: dim
  !
  logical :: parity
end function

attributes(intrinsic,std_f2008)&
elemental function popcnt(i)
  integer :: i
  !
  integer :: popcnt
end function

attributes(intrinsic,std_f2008)&
elemental function poppar(i)
  integer :: i
  !
  integer :: poppar
end function

attributes(intrinsic,std_f2008)&
elemental function shifta(i,shift)
  integer :: i
  integer :: shift
  !
  integer :: shifta
end function

attributes(intrinsic,std_f2008)&
elemental function shiftl(i,shift)
  integer :: i
  integer :: shift
  !
  integer :: shiftl
end function

attributes(intrinsic,std_f2008)&
elemental function shiftr(i,shift)
  integer :: i
  integer :: shift
  !
  integer :: shiftr
end function

attributes(intrinsic,inquiry,std_f2008)&
function c_sizeof(x)
  type(*), dimension(..) :: x
  !
  integer :: c_sizeof
end function

attributes(intrinsic,inquiry,std_f2008)&
function compiler_options()
  !
  character :: compiler_options
end function

attributes(intrinsic,inquiry,std_f2008)&
function compiler_version()
  !
  character :: compiler_version
end function

attributes(intrinsic,inquiry,std_f2008)&
function storage_size(a,kind)
  type(*), dimension(..) :: a
  integer, optional :: kind
  !
  integer :: storage_size
end function

attributes(intrinsic,inquiry,std_f2008)&
function this_image(coarray,dim,distance)
  real, optional :: coarray
  integer, optional :: dim
  integer, optional :: distance
  !
  integer :: this_image
end function

attributes(intrinsic,std_f2008)&
elemental function trailz(i)
  integer :: i
  !
  integer :: trailz
end function

attributes(intrinsic,inquiry,std_f2008)&
function ucobound(coarray,dim,kind)
  real :: coarray
  integer, optional :: dim
  integer, optional :: kind
  !
  integer :: ucobound
end function

attributes(intrinsic,atomic,std_f2008)&
subroutine atomic_define(atom,value,stat)
  integer, intent(out) :: atom
  integer, intent(in) :: value
  integer, intent(out), optional :: stat
end subroutine

attributes(intrinsic,atomic,std_f2008)&
subroutine atomic_ref(value,atom,stat)
  integer, intent(out) :: value
  integer, intent(in) :: atom
  integer, intent(out), optional :: stat
end subroutine

attributes(intrinsic,impure,std_f2008)&
subroutine execute_command_line(command,wait,exitstat,cmdstat,cmdmsg)
  character, intent(in) :: command
  logical, intent(in), optional :: wait
  integer, intent(inout), optional :: exitstat
  integer, intent(out), optional :: cmdstat
  character, intent(inout), optional :: cmdmsg
end subroutine

