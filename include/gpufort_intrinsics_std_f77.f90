! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
! This file solely exists to extract information about intrinsics via GPUFORT's indexer tool.
! It will not compile with any compiler as non-standard Fortran expressions are used.
! conversion indicates the optional kind argument
attributes(intrinsic,conversion,kind_arg,std_f77)& 
pure elemental function int(a,kind)
  type(*) :: a
  integer,optional :: kind
  !
  integer :: int
end function

attributes(intrinsic,conversion,kind_arg,std_f77)&
pure elemental function real(a,kind)
  type(*) :: a
  integer,optional :: kind
  !
  real :: real
end function

attributes(intrinsic,conversion,kind_arg,std_f77)&
pure elemental function cmplx(x,y,kind)
  type(*) :: x
  type(*),optional :: y
  integer,optional :: kind
  !
  complex :: cmplx
end function

attributes(intrinsic,std_f77,actual)&
elemental function abs(a)
  real :: a
  !
  real :: abs
end function

attributes(intrinsic,std_f77,actual)&
elemental function iabs(a)
  integer :: a
  !
  integer :: iabs
end function

attributes(intrinsic,std_f77,actual)&
elemental function dabs(a)
  double precision :: a
  !
  double precision :: dabs
end function

attributes(intrinsic,std_f77,actual)&
elemental function cabs(a)
  complex :: a
  !
  real :: cabs
end function

attributes(intrinsic,std_f77,actual)&
elemental function acos(x)
  real :: x
  !
  real :: acos
end function

attributes(intrinsic,std_f77,actual)&
elemental function dacos(x)
  double precision :: x
  !
  double precision :: dacos
end function

attributes(intrinsic,std_f77,actual)&
elemental function aimag(z)
  complex :: z
  !
  real :: aimag
end function

attributes(intrinsic,kind_arg,std_f77,actual)&
elemental function aint(a,kind)
  real :: a
  integer, optional :: kind
  !
  real :: aint
end function

attributes(intrinsic,std_f77,actual)&
elemental function dint(a)
  double precision :: a
  !
  double precision :: dint
end function

attributes(intrinsic,kind_arg,std_f77,actual)&
elemental function anint(a,kind)
  real :: a
  integer, optional :: kind
  !
  real :: anint
end function

attributes(intrinsic,std_f77,actual)&
elemental function dnint(a)
  double precision :: a
  !
  double precision :: dnint
end function

attributes(intrinsic,std_f77,actual)&
elemental function asin(x)
  real :: x
  !
  real :: asin
end function

attributes(intrinsic,std_f77,actual)&
elemental function dasin(x)
  double precision :: x
  !
  double precision :: dasin
end function

attributes(intrinsic,std_f77,actual)&
elemental function atan(x)
  real :: x
  !
  real :: atan
end function

attributes(intrinsic,std_f77,actual)&
elemental function datan(x)
  double precision :: x
  !
  double precision :: datan
end function

attributes(intrinsic,std_f77,actual)&
elemental function atan2(y,x)
  real :: y
  real :: x
  !
  real :: atan2
end function

attributes(intrinsic,std_f77,actual)&
elemental function datan2(y,x)
  double precision :: y
  double precision :: x
  !
  double precision :: datan2
end function

attributes(intrinsic,std_f77)&
elemental function char(i,kind)
  integer :: i
  integer, optional :: kind
  !
  character :: char
end function

attributes(intrinsic,std_f77,actual)&
elemental function conjg(z)
  complex :: z
  !
  complex :: conjg
end function

attributes(intrinsic,std_f77,actual)&
elemental function cos(x)
  real :: x
  !
  real :: cos
end function

attributes(intrinsic,std_f77,actual)&
elemental function dcos(x)
  double precision :: x
  !
  double precision :: dcos
end function

attributes(intrinsic,std_f77,actual)&
elemental function ccos(x)
  complex :: x
  !
  complex :: ccos
end function

attributes(intrinsic,std_f77,actual)&
elemental function cosh(x)
  real :: x
  !
  real :: cosh
end function

attributes(intrinsic,std_f77,actual)&
elemental function dcosh(x)
  double precision :: x
  !
  double precision :: dcosh
end function

attributes(intrinsic,std_f77)&
elemental function dble(a)
  real :: a
  !
  double precision :: dble
end function

attributes(intrinsic,std_f77,actual)&
elemental function dim(x,y)
  real :: x
  real :: y
  !
  real :: dim
end function

attributes(intrinsic,std_f77,actual)&
elemental function idim(x,y)
  integer :: x
  integer :: y
  !
  integer :: idim
end function

attributes(intrinsic,std_f77,actual)&
elemental function ddim(x,y)
  double precision :: x
  double precision :: y
  !
  double precision :: ddim
end function

attributes(intrinsic,std_f77,actual)&
elemental function dprod(x,y)
  real :: x
  real :: y
  !
  double precision :: dprod
end function

attributes(intrinsic,std_f77,actual)&
elemental function exp(x)
  real :: x
  !
  real :: exp
end function

attributes(intrinsic,std_f77,actual)&
elemental function dexp(x)
  double precision :: x
  !
  double precision :: dexp
end function

attributes(intrinsic,std_f77,actual)&
elemental function cexp(x)
  complex :: x
  !
  complex :: cexp
end function

attributes(intrinsic,kind_arg,std_f77)&
elemental function ichar(c,kind)
  character :: c
  integer, optional :: kind
  !
  integer :: ichar
end function

attributes(intrinsic,kind_arg,std_f77,actual)&
elemental function index(string,substring,back,kind)
  character :: string
  character :: substring
  logical, optional :: back
  integer, optional :: kind
  !
  integer :: index
end function

attributes(intrinsic,std_f77)&
elemental function ifix(a)
  real :: a
  !
  integer :: ifix
end function

attributes(intrinsic,std_f77)&
elemental function idint(a)
  double precision :: a
  !
  integer :: idint
end function

attributes(intrinsic,inquiry,std_f77,actual)&
function len(string,kind)
  character :: string
  integer, optional :: kind
  !
  integer :: len
end function

attributes(intrinsic,std_f77)&
elemental function lge(string_a,string_b)
  character :: string_a
  character :: string_b
  !
  logical :: lge
end function

attributes(intrinsic,std_f77)&
elemental function lgt(string_a,string_b)
  character :: string_a
  character :: string_b
  !
  logical :: lgt
end function

attributes(intrinsic,std_f77)&
elemental function lle(string_a,string_b)
  character :: string_a
  character :: string_b
  !
  logical :: lle
end function

attributes(intrinsic,std_f77)&
elemental function llt(string_a,string_b)
  character :: string_a
  character :: string_b
  !
  logical :: llt
end function

attributes(intrinsic,std_f77)&
elemental function log(x)
  real :: x
  !
  real :: log
end function

attributes(intrinsic,std_f77,actual)&
elemental function alog(x)
  real :: x
  !
  real :: alog
end function

attributes(intrinsic,std_f77,actual)&
elemental function dlog(x)
  double precision :: x
  !
  double precision :: dlog
end function

attributes(intrinsic,std_f77,actual)&
elemental function clog(x)
  complex :: x
  !
  complex :: clog
end function

attributes(intrinsic,std_f77)&
elemental function log10(x)
  real :: x
  !
  real :: log10
end function

attributes(intrinsic,std_f77,actual)&
elemental function alog10(x)
  real :: x
  !
  real :: alog10
end function

attributes(intrinsic,std_f77,actual)&
elemental function dlog10(x)
  double precision :: x
  !
  double precision :: dlog10
end function

attributes(intrinsic,std_f77)&
elemental function max(a1,a2)
  type(*), dimension(..) :: a1
  type(*), dimension(..) :: a2
  !
  type(*), dimension(..) :: max
end function

attributes(intrinsic,std_f77)&
elemental function max0(a1,a2)
  integer :: a1
  integer :: a2
  !
  integer :: max0
end function

attributes(intrinsic,std_f77)&
elemental function amax0(a1,a2)
  integer :: a1
  integer :: a2
  !
  real :: amax0
end function

attributes(intrinsic,std_f77)&
elemental function amax1(a1,a2)
  real :: a1
  real :: a2
  !
  real :: amax1
end function

attributes(intrinsic,std_f77)&
elemental function max1(a1,a2)
  real :: a1
  real :: a2
  !
  integer :: max1
end function

attributes(intrinsic,std_f77)&
elemental function dmax1(a1,a2)
  double precision :: a1
  double precision :: a2
  !
  double precision :: dmax1
end function

attributes(intrinsic,std_f77)&
elemental function min(a1,a2)
  real :: a1
  real :: a2
  !
  type(*), dimension(..) :: min
end function

attributes(intrinsic,std_f77)&
elemental function min0(a1,a2)
  integer :: a1
  integer :: a2
  !
  integer :: min0
end function

attributes(intrinsic,std_f77)&
elemental function amin0(a1,a2)
  integer :: a1
  integer :: a2
  !
  real :: amin0
end function

attributes(intrinsic,std_f77)&
elemental function amin1(a1,a2)
  real :: a1
  real :: a2
  !
  real :: amin1
end function

attributes(intrinsic,std_f77)&
elemental function min1(a1,a2)
  real :: a1
  real :: a2
  !
  integer :: min1
end function

attributes(intrinsic,std_f77)&
elemental function dmin1(a1,a2)
  double precision :: a1
  double precision :: a2
  !
  double precision :: dmin1
end function

attributes(intrinsic,std_f77,actual)&
elemental function mod(a,p)
  integer :: a
  integer :: p
  !
  integer :: mod
end function

attributes(intrinsic,std_f77,actual)&
elemental function amod(a,p)
  real :: a
  real :: p
  !
  real :: amod
end function

attributes(intrinsic,std_f77,actual)&
elemental function dmod(a,p)
  double precision :: a
  double precision :: p
  !
  double precision :: dmod
end function

attributes(intrinsic,kind_arg,std_f77,actual)&
elemental function nint(a,kind)
  real :: a
  integer, optional :: kind
  !
  integer :: nint
end function

attributes(intrinsic,std_f77,actual)&
elemental function idnint(a)
  double precision :: a
  !
  integer :: idnint
end function

attributes(intrinsic,std_f77)&
elemental function float(a)
  integer :: a
  !
  real :: float
end function

attributes(intrinsic,std_f77)&
elemental function sngl(a)
  double precision :: a
  !
  real :: sngl
end function

attributes(intrinsic,std_f77,actual)&
elemental function sign(a,b)
  real :: a
  real :: b
  !
  real :: sign
end function

attributes(intrinsic,std_f77,actual)&
elemental function isign(a,b)
  integer :: a
  integer :: b
  !
  integer :: isign
end function

attributes(intrinsic,std_f77,actual)&
elemental function dsign(a,b)
  double precision :: a
  double precision :: b
  !
  double precision :: dsign
end function

attributes(intrinsic,std_f77,actual)&
elemental function sin(x)
  real :: x
  !
  real :: sin
end function

attributes(intrinsic,std_f77,actual)&
elemental function dsin(x)
  double precision :: x
  !
  double precision :: dsin
end function

attributes(intrinsic,std_f77,actual)&
elemental function csin(x)
  complex :: x
  !
  complex :: csin
end function

attributes(intrinsic,std_f77,actual)&
elemental function sinh(x)
  real :: x
  !
  real :: sinh
end function

attributes(intrinsic,std_f77,actual)&
elemental function dsinh(x)
  double precision :: x
  !
  double precision :: dsinh
end function

attributes(intrinsic,std_f77,actual)&
elemental function sqrt(x)
  real :: x
  !
  real :: sqrt
end function

attributes(intrinsic,std_f77,actual)&
elemental function dsqrt(x)
  double precision :: x
  !
  double precision :: dsqrt
end function

attributes(intrinsic,std_f77,actual)&
elemental function csqrt(x)
  complex :: x
  !
  complex :: csqrt
end function

attributes(intrinsic,std_f77,actual)&
elemental function tan(x)
  real :: x
  !
  real :: tan
end function

attributes(intrinsic,std_f77,actual)&
elemental function dtan(x)
  double precision :: x
  !
  double precision :: dtan
end function

attributes(intrinsic,std_f77,actual)&
elemental function tanh(x)
  real :: x
  !
  real :: tanh
end function

attributes(intrinsic,std_f77,actual)&
elemental function dtanh(x)
  double precision :: x
  !
  double precision :: dtanh
end function