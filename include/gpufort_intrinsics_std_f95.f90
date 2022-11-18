! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
! This file solely exists to extract information about intrinsics via GPUFORT's indexer tool.
! It will not compile with any compiler as non-standard Fortran expressions are used.
attributes(intrinsic,conversion,kind_arg,std_f95)&
pure elemental function logical(x,kind)
  type(*) :: x
  integer,optional :: kind
  !
  logical :: logical
end function

attributes(intrinsic,std_f95)&
elemental function achar(i,kind)
  integer :: i
  integer, optional :: kind
  !
  character :: achar
end function

attributes(intrinsic,std_f95)&
elemental function adjustl(string)
  character :: string
  !
  character :: adjustl
end function

attributes(intrinsic,std_f95)&
elemental function adjustr(string)
  character :: string
  !
  character :: adjustr
end function

attributes(intrinsic,transformational,std_f95)&
function all(mask,dim)
  logical :: mask
  integer, optional :: dim
  !
  logical :: all
end function

attributes(intrinsic,inquiry,std_f95)&
function allocated(array)
  type(*), dimension(..) :: array
  !
  logical :: allocated
end function

attributes(intrinsic,transformational,std_f95)&
function any(mask,dim)
  logical :: mask
  integer, optional :: dim
  !
  logical :: any
end function

attributes(intrinsic,inquiry,std_f95)&
function associated(pointer,target)
  type(*), dimension(..) :: pointer
  type(*), dimension(..), optional :: target
  !
  logical :: associated
end function

attributes(intrinsic,inquiry,std_f95)&
function bit_size(i)
  integer :: i
  !
  integer :: bit_size
end function

attributes(intrinsic,std_f95)&
elemental function btest(i,pos)
  integer :: i
  integer :: pos
  !
  logical :: btest
end function

attributes(intrinsic,kind_arg,std_f95)&
elemental function ceiling(a,kind)
  real :: a
  integer, optional :: kind
  !
  integer :: ceiling
end function

attributes(intrinsic,transformational,std_f95)&
function count(mask,dim,kind)
  logical :: mask
  integer, optional :: dim
  integer, optional :: kind
  !
  integer :: count
end function

attributes(intrinsic,transformational,std_f95)&
function cshift(array,shift,dim)
  real :: array
  integer :: shift
  integer, optional :: dim
  !
  real :: cshift
end function

attributes(intrinsic,inquiry,std_f95)&
function digits(x)
  type(*), dimension(..) :: x
  !
  integer :: digits
end function

attributes(intrinsic,transformational,std_f95)&
function dot_product(vector_a,vector_b)
  real :: vector_a
  real :: vector_b
  !
  real :: dot_product
end function

attributes(intrinsic,transformational,std_f95)&
function eoshift(array,shift,boundary,dim)
  real :: array
  integer :: shift
  real, optional :: boundary
  integer, optional :: dim
  !
  real :: eoshift
end function

attributes(intrinsic,inquiry,std_f95)&
function epsilon(x)
  real :: x
  !
  real :: epsilon
end function

attributes(intrinsic,std_f95)&
elemental function exponent(x)
  real :: x
  !
  integer :: exponent
end function

attributes(intrinsic,kind_arg,std_f95)&
elemental function floor(a,kind)
  real :: a
  integer, optional :: kind
  !
  integer :: floor
end function

attributes(intrinsic,std_f95)&
elemental function fraction(x)
  real :: x
  !
  real :: fraction
end function

attributes(intrinsic,inquiry,std_f95)&
function huge(x)
  type(*), dimension(..) :: x
  !
  real :: huge
end function

attributes(intrinsic,kind_arg,std_f95)&
elemental function iachar(c,kind)
  character :: c
  integer, optional :: kind
  !
  integer :: iachar
end function

attributes(intrinsic,std_f95)&
elemental function iand(i,j)
  integer :: i
  integer :: j
  !
  integer :: iand
end function

attributes(intrinsic,std_f95)&
elemental function ibclr(i,pos)
  integer :: i
  integer :: pos
  !
  integer :: ibclr
end function

attributes(intrinsic,std_f95)&
elemental function ibits(i,pos,len)
  integer :: i
  integer :: pos
  integer :: len
  !
  integer :: ibits
end function

attributes(intrinsic,std_f95)&
elemental function ibset(i,pos)
  integer :: i
  integer :: pos
  !
  integer :: ibset
end function

attributes(intrinsic,std_f95)&
elemental function ieor(i,j)
  integer :: i
  integer :: j
  !
  integer :: ieor
end function

attributes(intrinsic,std_f95)&
elemental function ior(i,j)
  integer :: i
  integer :: j
  !
  integer :: ior
end function

attributes(intrinsic,std_f95)&
elemental function ishft(i,shift)
  integer :: i
  integer :: shift
  !
  integer :: ishft
end function

attributes(intrinsic,std_f95)&
elemental function ishftc(i,shift,size)
  integer :: i
  integer :: shift
  integer, optional :: size
  !
  integer :: ishftc
end function

attributes(intrinsic,inquiry,std_f95)&
function kind(x)
  real :: x
  !
  integer :: kind
end function

attributes(intrinsic,inquiry,std_f95)&
function lbound(array,dim,kind)
  real :: array
  integer, optional :: dim
  integer, optional :: kind
  !
  integer :: lbound
end function

attributes(intrinsic,kind_arg,std_f95)&
elemental function len_trim(string,kind)
  character :: string
  integer, optional :: kind
  !
  integer :: len_trim
end function

attributes(intrinsic,transformational,std_f95)&
function matmul(matrix_a,matrix_b)
  real :: matrix_a
  real :: matrix_b
  !
  real :: matmul
end function

attributes(intrinsic,inquiry,std_f95)&
function maxexponent(x)
  type(*), dimension(..) :: x
  !
  integer :: maxexponent
end function

attributes(intrinsic,transformational,std_f95)&
function maxloc(array,dim,mask,kind,back)
  real :: array
  integer, optional :: dim
  logical, optional :: mask
  integer, optional :: kind
  logical, optional :: back
  !
  integer :: maxloc
end function

attributes(intrinsic,transformational,std_f95)&
function maxval(array,dim,mask)
  real :: array
  integer, optional :: dim
  logical, optional :: mask
  !
  real :: maxval
end function

attributes(intrinsic,std_f95)&
elemental function merge(tsource,fsource,mask)
  real :: tsource
  real :: fsource
  logical :: mask
  !
  real :: merge
end function

attributes(intrinsic,inquiry,std_f95)&
function minexponent(x)
  type(*), dimension(..) :: x
  !
  integer :: minexponent
end function

attributes(intrinsic,transformational,std_f95)&
function minloc(array,dim,mask,kind,back)
  real :: array
  integer, optional :: dim
  logical, optional :: mask
  integer, optional :: kind
  logical, optional :: back
  !
  integer :: minloc
end function

attributes(intrinsic,transformational,std_f95)&
function minval(array,dim,mask)
  real :: array
  integer, optional :: dim
  logical, optional :: mask
  !
  real :: minval
end function

attributes(intrinsic,std_f95)&
elemental function modulo(a,p)
  real :: a
  real :: p
  !
  real :: modulo
end function

attributes(intrinsic,std_f95)&
elemental function nearest(x,s)
  real :: x
  real :: s
  !
  real :: nearest
end function

attributes(intrinsic,std_f95)&
elemental function not(i)
  integer :: i
  !
  integer :: not
end function

attributes(intrinsic,transformational,std_f95)&
function null(mold)
  integer, optional :: mold
  !
  integer :: null
end function

attributes(intrinsic,transformational,std_f95)&
function pack(array,mask,vector)
  real :: array
  logical :: mask
  real, optional :: vector
  !
  real :: pack
end function

attributes(intrinsic,inquiry,std_f95)&
function precision(x)
  type(*), dimension(..) :: x
  !
  integer :: precision
end function

attributes(intrinsic,inquiry,std_f95)&
function present(a)
  real :: a
  !
  logical :: present
end function

attributes(intrinsic,transformational,std_f95)&
function product(array,dim,mask)
  real :: array
  integer, optional :: dim
  logical, optional :: mask
  !
  real :: product
end function

attributes(intrinsic,inquiry,std_f95)&
function radix(x)
  type(*), dimension(..) :: x
  !
  integer :: radix
end function

attributes(intrinsic,inquiry,std_f95)&
function range(x)
  real :: x
  !
  integer :: range
end function

attributes(intrinsic,transformational,std_f95)&
function repeat(string,ncopies)
  character :: string
  integer :: ncopies
  !
  character :: repeat
end function

attributes(intrinsic,transformational,std_f95)&
function reshape(source,shape,pad,order)
  real :: source
  integer :: shape
  real, optional :: pad
  integer, optional :: order
  !
  real :: reshape
end function

attributes(intrinsic,std_f95)&
elemental function rrspacing(x)
  real :: x
  !
  real :: rrspacing
end function

attributes(intrinsic,std_f95)&
elemental function scale(x,i)
  real :: x
  integer :: i
  !
  real :: scale
end function

attributes(intrinsic,kind_arg,std_f95)&
elemental function scan(string,set,back,kind)
  character :: string
  character :: set
  logical, optional :: back
  integer, optional :: kind
  !
  integer :: scan
end function

attributes(intrinsic,transformational,std_f95)&
function selected_int_kind(r)
  integer :: r
  !
  integer :: selected_int_kind
end function

attributes(intrinsic,transformational,std_f95)&
function selected_real_kind(p,r,radix)
  integer, optional :: p
  integer, optional :: r
  integer, optional :: radix
  !
  integer :: selected_real_kind
end function

attributes(intrinsic,std_f95)&
elemental function set_exponent(x,i)
  real :: x
  integer :: i
  !
  real :: set_exponent
end function

attributes(intrinsic,inquiry,std_f95)&
function shape(source,kind)
  real :: source
  integer, optional :: kind
  !
  integer :: shape
end function

attributes(intrinsic,inquiry,std_f95)&
function size(array,dim,kind)
  real :: array
  integer, optional :: dim
  integer, optional :: kind
  !
  integer :: size
end function

attributes(intrinsic,std_f95)&
elemental function spacing(x)
  real :: x
  !
  real :: spacing
end function

attributes(intrinsic,transformational,std_f95)&
function spread(source,dim,ncopies)
  real :: source
  integer :: dim
  integer :: ncopies
  !
  real :: spread
end function

attributes(intrinsic,transformational,std_f95)&
function sum(array,dim,mask)
  real :: array
  integer, optional :: dim
  logical, optional :: mask
  !
  real :: sum
end function

attributes(intrinsic,inquiry,std_f95)&
function tiny(x)
  real :: x
  !
  real :: tiny
end function

attributes(intrinsic,transformational,std_f95)&
function transfer(source,mold,size)
  real :: source
  real :: mold
  integer, optional :: size
  !
  real :: transfer
end function

attributes(intrinsic,transformational,std_f95)&
function transpose(matrix)
  real :: matrix
  !
  real :: transpose
end function

attributes(intrinsic,transformational,std_f95)&
function trim(string)
  character :: string
  !
  character :: trim
end function

attributes(intrinsic,inquiry,std_f95)&
function ubound(array,dim,kind)
  real :: array
  integer, optional :: dim
  integer, optional :: kind
  !
  integer :: ubound
end function

attributes(intrinsic,transformational,std_f95)&
function unpack(vector,mask,field)
  real :: vector
  logical :: mask
  real :: field
  !
  real :: unpack
end function

attributes(intrinsic,kind_arg,std_f95)&
elemental function verify(string,set,back,kind)
  character :: string
  character :: set
  logical, optional :: back
  integer, optional :: kind
  !
  integer :: verify
end function

attributes(intrinsic,impure,std_f95)&
subroutine cpu_time(time)
  real, intent(out) :: time
end subroutine

attributes(intrinsic,impure,std_f95)&
subroutine date_and_time(date,time,zone,values)
  character, intent(out), optional :: date
  character, intent(out), optional :: time
  character, intent(out), optional :: zone
  integer, intent(out), optional :: values
end subroutine

attributes(intrinsic,std_f95)&
elemental subroutine mvbits(from,frompos,len,to,topos)
  integer, intent(in) :: from
  integer, intent(in) :: frompos
  integer, intent(in) :: len
  integer, intent(inout) :: to
  integer, intent(in) :: topos
end subroutine

attributes(intrinsic,impure,std_f95)&
subroutine random_number(harvest)
  real, intent(out) :: harvest
end subroutine

attributes(intrinsic,impure,std_f95)&
subroutine random_seed(size,put,get)
  integer, intent(out), optional :: size
  integer, intent(in), optional :: put
  integer, intent(out), optional :: get
end subroutine

attributes(intrinsic,impure,std_f95)&
subroutine system_clock(count,count_rate,count_max)
  integer, intent(out), optional :: count
  integer, intent(out), optional :: count_rate
  integer, intent(out), optional :: count_max
end subroutine
