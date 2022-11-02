! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
! This file solely exists to extract information about intrinsics via GPUFORT's indexer tool.
! It will not compile with any compiler as non-standard Fortran expressions are used.
attributes(intrinsic,std_gnu,actual)&
elemental function kiabs(a)
  real :: a
  !
  real :: kiabs
end function

attributes(intrinsic,std_gnu,actual)&
elemental function jiabs(a)
  real :: a
  !
  real :: jiabs
end function

attributes(intrinsic,std_gnu,actual)&
elemental function iiabs(a)
  real :: a
  !
  real :: iiabs
end function

attributes(intrinsic,std_gnu,actual)&
elemental function babs(a)
  real :: a
  !
  real :: babs
end function

attributes(intrinsic,std_gnu,actual)&
elemental function zabs(a)
  double complex :: a
  !
  double precision :: zabs
end function

attributes(intrinsic,std_gnu,actual)&
elemental function cdabs(a)
  double complex :: a
  !
  double precision :: cdabs
end function

attributes(intrinsic,impure,std_gnu)&
function access(name,mode)
  character :: name
  character :: mode
  !
  integer :: access
end function

attributes(intrinsic,std_gnu,actual)&
elemental function dacosh(x)
  double precision :: x
  !
  double precision :: dacosh
end function

attributes(intrinsic,std_gnu,actual)&
elemental function imagpart(z)
  complex :: z
  !
  real :: imagpart
end function

attributes(intrinsic,std_gnu,actual)&
elemental function imag(z)
  complex :: z
  !
  real :: imag
end function

attributes(intrinsic,std_gnu,actual)&
elemental function dimag(z)
  double complex :: z
  !
  double precision :: dimag
end function

attributes(intrinsic,std_gnu,actual)&
elemental function dasinh(x)
  double precision :: x
  !
  double precision :: dasinh
end function

attributes(intrinsic,std_gnu,actual)&
elemental function datanh(x)
  double precision :: x
  !
  double precision :: datanh
end function

attributes(intrinsic,std_gnu)&
elemental function besj0(x)
  real :: x
  !
  real :: besj0
end function

attributes(intrinsic,std_gnu)&
elemental function dbesj0(x)
  double precision :: x
  !
  double precision :: dbesj0
end function

attributes(intrinsic,std_gnu)&
elemental function besj1(x)
  real :: x
  !
  real :: besj1
end function

attributes(intrinsic,std_gnu)&
elemental function dbesj1(x)
  double precision :: x
  !
  double precision :: dbesj1
end function

attributes(intrinsic,std_gnu)&
elemental function besjn(n,x)
  integer :: n
  real :: x
  !
  real :: besjn
end function

attributes(intrinsic,std_gnu)&
elemental function dbesjn(n,x)
  integer :: n
  double precision :: x
  !
  double precision :: dbesjn
end function

attributes(intrinsic,std_gnu)&
elemental function besy0(x)
  real :: x
  !
  real :: besy0
end function

attributes(intrinsic,std_gnu)&
elemental function dbesy0(x)
  double precision :: x
  !
  double precision :: dbesy0
end function

attributes(intrinsic,std_gnu)&
elemental function besy1(x)
  real :: x
  !
  real :: besy1
end function

attributes(intrinsic,std_gnu)&
elemental function dbesy1(x)
  double precision :: x
  !
  double precision :: dbesy1
end function

attributes(intrinsic,std_gnu)&
elemental function besyn(n,x)
  integer :: n
  real :: x
  !
  real :: besyn
end function

attributes(intrinsic,std_gnu)&
elemental function dbesyn(n,x)
  integer :: n
  double precision :: x
  !
  double precision :: dbesyn
end function

attributes(intrinsic,std_gnu)&
elemental function bktest(i,pos)
  integer :: i
  integer :: pos
  !
  logical :: bktest
end function

attributes(intrinsic,std_gnu)&
elemental function bjtest(i,pos)
  integer :: i
  integer :: pos
  !
  logical :: bjtest
end function

attributes(intrinsic,std_gnu)&
elemental function bitest(i,pos)
  integer :: i
  integer :: pos
  !
  logical :: bitest
end function

attributes(intrinsic,std_gnu)&
elemental function bbtest(i,pos)
  integer :: i
  integer :: pos
  !
  logical :: bbtest
end function

attributes(intrinsic,impure,std_gnu)&
function chdir(name)
  character :: name
  !
  integer :: chdir
end function

attributes(intrinsic,impure,std_gnu)&
function chmod(name,mode)
  character :: name
  character :: mode
  !
  integer :: chmod
end function

attributes(intrinsic,std_gnu)&
elemental function complex(x,y)
  type(*), dimension(..) :: x
  type(*), dimension(..) :: y
  !
  complex :: complex
end function

attributes(intrinsic,std_gnu)&
elemental function dcmplx(x,y)
  double precision :: x
  double precision, optional :: y
  !
  double complex :: dcmplx
end function

attributes(intrinsic,std_gnu,actual)&
elemental function dconjg(z)
  double complex :: z
  !
  double complex :: dconjg
end function

attributes(intrinsic,std_gnu,actual)&
elemental function zcos(x)
  double complex :: x
  !
  double complex :: zcos
end function

attributes(intrinsic,std_gnu,actual)&
elemental function cdcos(x)
  double complex :: x
  !
  double complex :: cdcos
end function

attributes(intrinsic,impure,std_gnu)&
function ctime(time)
  integer :: time
  !
  character :: ctime
end function

attributes(intrinsic,std_gnu)&
elemental function dreal(a)
  double complex :: a
  !
  double precision :: dreal
end function

attributes(intrinsic,std_gnu)&
elemental function derf(x)
  double precision :: x
  !
  double precision :: derf
end function

attributes(intrinsic,std_gnu)&
elemental function derfc(x)
  double precision :: x
  !
  double precision :: derfc
end function

attributes(intrinsic,impure,std_gnu)&
function dtime(x)
  real(4) :: x
  !
  real(4) :: dtime
end function

attributes(intrinsic,impure,std_gnu)&
function etime(x)
  real(4) :: x
  !
  real(4) :: etime
end function

attributes(intrinsic,std_gnu,actual)&
elemental function zexp(x)
  double complex :: x
  !
  double complex :: zexp
end function

attributes(intrinsic,std_gnu,actual)&
elemental function cdexp(x)
  double complex :: x
  !
  double complex :: cdexp
end function

attributes(intrinsic,impure,std_gnu)&
function fdate()
  !
  character :: fdate
end function

attributes(intrinsic,impure,std_gnu)&
function fnum(unit)
  integer :: unit
  !
  integer :: fnum
end function

attributes(intrinsic,impure,std_gnu)&
function fstat(unit,values)
  integer, intent(in) :: unit
  integer, intent(out) :: values
  !
  integer :: fstat
end function

attributes(intrinsic,impure,std_gnu)&
function ftell(unit)
  integer :: unit
  !
  integer :: ftell
end function

attributes(intrinsic,impure,std_gnu)&
function fgetc(unit,c)
  integer, intent(in) :: unit
  character, intent(out) :: c
  !
  integer :: fgetc
end function

attributes(intrinsic,impure,std_gnu)&
function fget(c)
  character, intent(out) :: c
  !
  integer :: fget
end function

attributes(intrinsic,impure,std_gnu)&
function fputc(unit,c)
  integer :: unit
  character :: c
  !
  integer :: fputc
end function

attributes(intrinsic,impure,std_gnu)&
function fput(c)
  character :: c
  !
  integer :: fput
end function

attributes(intrinsic,std_gnu)&
elemental function dgamma(x)
  real :: x
  !
  real :: dgamma
end function

attributes(intrinsic,impure,std_gnu)&
function getcwd(c)
  character :: c
  !
  integer :: getcwd
end function

attributes(intrinsic,impure,std_gnu)&
function getgid()
  !
  integer :: getgid
end function

attributes(intrinsic,impure,std_gnu)&
function getpid()
  !
  integer :: getpid
end function

attributes(intrinsic,impure,std_gnu)&
function getuid()
  !
  integer :: getuid
end function

attributes(intrinsic,impure,std_gnu)&
function hostnm(c)
  character, intent(out) :: c
  !
  integer :: hostnm
end function

attributes(intrinsic,std_gnu)&
elemental function kiand(i,j)
  integer :: i
  integer :: j
  !
  integer :: kiand
end function

attributes(intrinsic,std_gnu)&
elemental function jiand(i,j)
  integer :: i
  integer :: j
  !
  integer :: jiand
end function

attributes(intrinsic,std_gnu)&
elemental function iiand(i,j)
  integer :: i
  integer :: j
  !
  integer :: iiand
end function

attributes(intrinsic,std_gnu)&
elemental function biand(i,j)
  integer :: i
  integer :: j
  !
  integer :: biand
end function

attributes(intrinsic,impure,std_gnu)&
function and(i,j)
  type(*), dimension(..) :: i
  type(*), dimension(..) :: j
  !
  logical :: and
end function

attributes(intrinsic,impure,std_gnu)&
function iargc()
  !
  integer :: iargc
end function

attributes(intrinsic,std_gnu)&
elemental function kibclr(i,pos)
  integer :: i
  integer :: pos
  !
  integer :: kibclr
end function

attributes(intrinsic,std_gnu)&
elemental function jibclr(i,pos)
  integer :: i
  integer :: pos
  !
  integer :: jibclr
end function

attributes(intrinsic,std_gnu)&
elemental function iibclr(i,pos)
  integer :: i
  integer :: pos
  !
  integer :: iibclr
end function

attributes(intrinsic,std_gnu)&
elemental function bbclr(i,pos)
  integer :: i
  integer :: pos
  !
  integer :: bbclr
end function

attributes(intrinsic,std_gnu)&
elemental function kibits(i,pos,len)
  integer :: i
  integer :: pos
  integer :: len
  !
  integer :: kibits
end function

attributes(intrinsic,std_gnu)&
elemental function jibits(i,pos,len)
  integer :: i
  integer :: pos
  integer :: len
  !
  integer :: jibits
end function

attributes(intrinsic,std_gnu)&
elemental function iibits(i,pos,len)
  integer :: i
  integer :: pos
  integer :: len
  !
  integer :: iibits
end function

attributes(intrinsic,std_gnu)&
elemental function bbits(i,pos,len)
  integer :: i
  integer :: pos
  integer :: len
  !
  integer :: bbits
end function

attributes(intrinsic,std_gnu)&
elemental function kibset(i,pos)
  integer :: i
  integer :: pos
  !
  integer :: kibset
end function

attributes(intrinsic,std_gnu)&
elemental function jibset(i,pos)
  integer :: i
  integer :: pos
  !
  integer :: jibset
end function

attributes(intrinsic,std_gnu)&
elemental function iibset(i,pos)
  integer :: i
  integer :: pos
  !
  integer :: iibset
end function

attributes(intrinsic,std_gnu)&
elemental function bbset(i,pos)
  integer :: i
  integer :: pos
  !
  integer :: bbset
end function

attributes(intrinsic,std_gnu)&
elemental function kieor(i,j)
  integer :: i
  integer :: j
  !
  integer :: kieor
end function

attributes(intrinsic,std_gnu)&
elemental function jieor(i,j)
  integer :: i
  integer :: j
  !
  integer :: jieor
end function

attributes(intrinsic,std_gnu)&
elemental function iieor(i,j)
  integer :: i
  integer :: j
  !
  integer :: iieor
end function

attributes(intrinsic,std_gnu)&
elemental function bieor(i,j)
  integer :: i
  integer :: j
  !
  integer :: bieor
end function

attributes(intrinsic,impure,std_gnu)&
function xor(i,j)
  type(*), dimension(..) :: i
  type(*), dimension(..) :: j
  !
  logical :: xor
end function

attributes(intrinsic,impure,std_gnu)&
function ierrno()
  !
  integer :: ierrno
end function

attributes(intrinsic,std_gnu)&
elemental function int2(a)
  real :: a
  !
  integer :: int2
end function

attributes(intrinsic,std_gnu)&
elemental function short(a)
  real :: a
  !
  integer :: short
end function

attributes(intrinsic,std_gnu)&
elemental function int8(a)
  real :: a
  !
  integer :: int8
end function

attributes(intrinsic,std_gnu)&
elemental function long(a)
  real :: a
  !
  integer :: long
end function

attributes(intrinsic,std_gnu)&
elemental function kior(i,j)
  integer :: i
  integer :: j
  !
  integer :: kior
end function

attributes(intrinsic,std_gnu)&
elemental function jior(i,j)
  integer :: i
  integer :: j
  !
  integer :: jior
end function

attributes(intrinsic,std_gnu)&
elemental function iior(i,j)
  integer :: i
  integer :: j
  !
  integer :: iior
end function

attributes(intrinsic,std_gnu)&
elemental function bior(i,j)
  integer :: i
  integer :: j
  !
  integer :: bior
end function

attributes(intrinsic,impure,std_gnu)&
function or(i,j)
  type(*), dimension(..) :: i
  type(*), dimension(..) :: j
  !
  logical :: or
end function

attributes(intrinsic,impure,std_gnu)&
function irand(i)
  integer(4), optional :: i
  !
  integer(4) :: irand
end function

attributes(intrinsic,impure,std_gnu)&
function isatty(unit)
  integer :: unit
  !
  logical :: isatty
end function

attributes(intrinsic,std_gnu)&
elemental function isnan(x)
  real :: x
  !
  logical :: isnan
end function

attributes(intrinsic,std_gnu)&
elemental function rshift(i,shift)
  integer :: i
  integer :: shift
  !
  integer :: rshift
end function

attributes(intrinsic,std_gnu)&
elemental function lshift(i,shift)
  integer :: i
  integer :: shift
  !
  integer :: lshift
end function

attributes(intrinsic,std_gnu)&
elemental function kishft(i,shift)
  integer :: i
  integer :: shift
  !
  integer :: kishft
end function

attributes(intrinsic,std_gnu)&
elemental function jishft(i,shift)
  integer :: i
  integer :: shift
  !
  integer :: jishft
end function

attributes(intrinsic,std_gnu)&
elemental function iishft(i,shift)
  integer :: i
  integer :: shift
  !
  integer :: iishft
end function

attributes(intrinsic,std_gnu)&
elemental function bshft(i,shift)
  integer :: i
  integer :: shift
  !
  integer :: bshft
end function

attributes(intrinsic,std_gnu)&
elemental function kishftc(i,shift,size)
  integer :: i
  integer :: shift
  integer, optional :: size
  !
  integer :: kishftc
end function

attributes(intrinsic,std_gnu)&
elemental function jishftc(i,shift,size)
  integer :: i
  integer :: shift
  integer, optional :: size
  !
  integer :: jishftc
end function

attributes(intrinsic,std_gnu)&
elemental function iishftc(i,shift,size)
  integer :: i
  integer :: shift
  integer, optional :: size
  !
  integer :: iishftc
end function

attributes(intrinsic,std_gnu)&
elemental function bshftc(i,shift,size)
  integer :: i
  integer :: shift
  integer, optional :: size
  !
  integer :: bshftc
end function

attributes(intrinsic,impure,std_gnu)&
function kill(pid,sig)
  integer :: pid
  integer :: sig
  !
  integer :: kill
end function

attributes(intrinsic,std_gnu)&
elemental function lnblnk(string,kind)
  character :: string
  integer, optional :: kind
  !
  integer :: lnblnk
end function

attributes(intrinsic,std_gnu)&
elemental function lgamma(x)
  real :: x
  !
  real :: lgamma
end function

attributes(intrinsic,std_gnu)&
elemental function algama(x)
  real :: x
  !
  real :: algama
end function

attributes(intrinsic,std_gnu)&
elemental function dlgama(x)
  real :: x
  !
  real :: dlgama
end function

attributes(intrinsic,impure,std_gnu)&
function link(path1,path2)
  character :: path1
  character :: path2
  !
  integer :: link
end function

attributes(intrinsic,std_gnu,actual)&
elemental function zlog(x)
  double complex :: x
  !
  double complex :: zlog
end function

attributes(intrinsic,std_gnu,actual)&
elemental function cdlog(x)
  double complex :: x
  !
  double complex :: cdlog
end function

attributes(intrinsic,impure,std_gnu)&
function lstat(name,values)
  character, intent(in) :: name
  integer, intent(out) :: values
  !
  integer :: lstat
end function

attributes(intrinsic,impure,std_gnu)&
function malloc(size)
  integer :: size
  !
  integer :: malloc
end function

attributes(intrinsic,impure,std_gnu)&
function mclock()
  !
  integer :: mclock
end function

attributes(intrinsic,impure,std_gnu)&
function mclock8()
  !
  integer :: mclock8
end function

attributes(intrinsic,std_gnu,actual)&
elemental function kmod(a,p)
  integer :: a
  integer :: p
  !
  integer :: kmod
end function

attributes(intrinsic,std_gnu,actual)&
elemental function jmod(a,p)
  integer :: a
  integer :: p
  !
  integer :: jmod
end function

attributes(intrinsic,std_gnu,actual)&
elemental function imod(a,p)
  integer :: a
  integer :: p
  !
  integer :: imod
end function

attributes(intrinsic,std_gnu,actual)&
elemental function bmod(a,p)
  integer :: a
  integer :: p
  !
  integer :: bmod
end function

attributes(intrinsic,std_gnu)&
elemental function knot(i)
  integer :: i
  !
  integer :: knot
end function

attributes(intrinsic,std_gnu)&
elemental function jnot(i)
  integer :: i
  !
  integer :: jnot
end function

attributes(intrinsic,std_gnu)&
elemental function inot(i)
  integer :: i
  !
  integer :: inot
end function

attributes(intrinsic,std_gnu)&
elemental function bnot(i)
  integer :: i
  !
  integer :: bnot
end function

attributes(intrinsic,impure,std_gnu)&
function rand(i)
  integer(4), optional :: i
  !
  real(4) :: rand
end function

attributes(intrinsic,impure,std_gnu)&
function ran(i)
  integer(4), optional :: i
  !
  real(4) :: ran
end function

attributes(intrinsic,std_gnu)&
elemental function realpart(a)
  type(*), dimension(..) :: a
  !
  real :: realpart
end function

attributes(intrinsic,std_gnu)&
elemental function floatk(a)
  integer :: a
  !
  real :: floatk
end function

attributes(intrinsic,std_gnu)&
elemental function floatj(a)
  integer :: a
  !
  real :: floatj
end function

attributes(intrinsic,std_gnu)&
elemental function floati(a)
  integer :: a
  !
  real :: floati
end function

attributes(intrinsic,std_gnu)&
elemental function dfloat(a)
  real :: a
  !
  double precision :: dfloat
end function

attributes(intrinsic,impure,std_gnu)&
function rename(path1,path2)
  character :: path1
  character :: path2
  !
  integer :: rename
end function

attributes(intrinsic,impure,std_gnu)&
function second()
  !
  real(4) :: second
end function

attributes(intrinsic,impure,std_gnu)&
function secnds(x)
  real :: x
  !
  real :: secnds
end function

attributes(intrinsic,impure,std_gnu)&
function signal(number,handler)
  integer :: number
  type(c_ptr) :: handler
  !
  integer :: signal
end function

attributes(intrinsic,std_gnu,actual)&
elemental function zsin(x)
  double complex :: x
  !
  double complex :: zsin
end function

attributes(intrinsic,std_gnu,actual)&
elemental function cdsin(x)
  double complex :: x
  !
  double complex :: cdsin
end function

attributes(intrinsic,inquiry,std_gnu)&
function sizeof(x)
  type(*), dimension(..) :: x
  !
  integer :: sizeof
end function

attributes(intrinsic,std_gnu,actual)&
elemental function zsqrt(x)
  double complex :: x
  !
  double complex :: zsqrt
end function

attributes(intrinsic,std_gnu,actual)&
elemental function cdsqrt(x)
  double complex :: x
  !
  double complex :: cdsqrt
end function

attributes(intrinsic,impure,std_gnu)&
function stat(name,values)
  character, intent(in) :: name
  integer, intent(out) :: values
  !
  integer :: stat
end function

attributes(intrinsic,impure,std_gnu)&
function symlnk(path1,path2)
  character :: path1
  character :: path2
  !
  integer :: symlnk
end function

attributes(intrinsic,impure,std_gnu)&
function system(command)
  character :: command
  !
  integer :: system
end function

attributes(intrinsic,impure,std_gnu)&
function time()
  !
  integer :: time
end function

attributes(intrinsic,impure,std_gnu)&
function time8()
  !
  integer :: time8
end function

attributes(intrinsic,impure,std_gnu)&
function ttynam(unit)
  integer :: unit
  !
  character :: ttynam
end function

attributes(intrinsic,impure,std_gnu)&
function umask(mask)
  integer :: mask
  !
  integer :: umask
end function

attributes(intrinsic,impure,std_gnu)&
function unlink(path)
  character :: path
  !
  integer :: unlink
end function

attributes(intrinsic,impure,std_gnu)&
function loc(x)
  type(*), dimension(..) :: x
  !
  integer :: loc
end function

attributes(intrinsic,std_gnu,actual)&
elemental function acosd(x)
  real :: x
  !
  real :: acosd
end function

attributes(intrinsic,std_gnu,actual)&
elemental function dacosd(x)
  double precision :: x
  !
  double precision :: dacosd
end function

attributes(intrinsic,std_gnu,actual)&
elemental function asind(x)
  real :: x
  !
  real :: asind
end function

attributes(intrinsic,std_gnu,actual)&
elemental function dasind(x)
  double precision :: x
  !
  double precision :: dasind
end function

attributes(intrinsic,std_gnu,actual)&
elemental function atand(x)
  real :: x
  !
  real :: atand
end function

attributes(intrinsic,std_gnu,actual)&
elemental function datand(x)
  double precision :: x
  !
  double precision :: datand
end function

attributes(intrinsic,std_gnu,actual)&
elemental function atan2d(y,x)
  real :: y
  real :: x
  !
  real :: atan2d
end function

attributes(intrinsic,std_gnu,actual)&
elemental function datan2d(y,x)
  double precision :: y
  double precision :: x
  !
  double precision :: datan2d
end function

attributes(intrinsic,std_gnu,actual)&
elemental function cosd(x)
  real :: x
  !
  real :: cosd
end function

attributes(intrinsic,std_gnu,actual)&
elemental function dcosd(x)
  double precision :: x
  !
  double precision :: dcosd
end function

attributes(intrinsic,std_gnu,actual)&
elemental function cotan(x)
  real :: x
  !
  real :: cotan
end function

attributes(intrinsic,std_gnu,actual)&
elemental function dcotan(x)
  double precision :: x
  !
  double precision :: dcotan
end function

attributes(intrinsic,std_gnu,actual)&
elemental function ccotan(x)
  complex :: x
  !
  complex :: ccotan
end function

attributes(intrinsic,std_gnu,actual)&
elemental function zcotan(x)
  double complex :: x
  !
  double complex :: zcotan
end function

attributes(intrinsic,std_gnu,actual)&
elemental function cotand(x)
  real :: x
  !
  real :: cotand
end function

attributes(intrinsic,std_gnu,actual)&
elemental function dcotand(x)
  double precision :: x
  !
  double precision :: dcotand
end function

attributes(intrinsic,std_gnu,actual)&
elemental function sind(x)
  real :: x
  !
  real :: sind
end function

attributes(intrinsic,std_gnu,actual)&
elemental function dsind(x)
  double precision :: x
  !
  double precision :: dsind
end function

attributes(intrinsic,std_gnu,actual)&
elemental function tand(x)
  real :: x
  !
  real :: tand
end function

attributes(intrinsic,std_gnu,actual)&
elemental function dtand(x)
  double precision :: x
  !
  double precision :: dtand
end function

attributes(intrinsic,None,std_gnu)&
subroutine abort()
end subroutine

attributes(intrinsic,None,std_gnu)&
subroutine backtrace()
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine ctime(time,result)
  integer, intent(in) :: time
  character, intent(out) :: result
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine idate(values)
  integer(4), intent(out) :: values
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine itime(values)
  integer(4), intent(out) :: values
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine ltime(time,values)
  integer, intent(in) :: time
  integer, intent(out) :: values
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine gmtime(time,values)
  integer, intent(in) :: time
  integer, intent(out) :: values
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine second(time)
  real, intent(out) :: time
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine chdir(name,status)
  character, intent(in) :: name
  integer, intent(out), optional :: status
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine chmod(name,mode,status)
  character, intent(in) :: name
  character, intent(in) :: mode
  integer, intent(out), optional :: status
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine etime(values,time)
  real(4), intent(out) :: values
  real(4), intent(out) :: time
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine dtime(values,time)
  real(4), intent(out) :: values
  real(4), intent(out) :: time
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine fdate(date)
  character, intent(out) :: date
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine gerror(result)
  character, intent(out) :: result
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine getcwd(count,status)
  character, intent(out) :: count
  integer, intent(out), optional :: status
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine getenv(name,value)
  character, intent(in) :: name
  character, intent(out) :: value
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine getarg(pos,value)
  integer, intent(in) :: pos
  character, intent(out) :: value
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine getlog(count)
  character, intent(out) :: count
end subroutine

attributes(intrinsic,std_gnu)&
elemental subroutine kmvbits(from,frompos,len,to,topos)
  integer, intent(in) :: from
  integer, intent(in) :: frompos
  integer, intent(in) :: len
  integer, intent(inout) :: to
  integer, intent(in) :: topos
end subroutine

attributes(intrinsic,std_gnu)&
elemental subroutine jmvbits(from,frompos,len,to,topos)
  integer, intent(in) :: from
  integer, intent(in) :: frompos
  integer, intent(in) :: len
  integer, intent(inout) :: to
  integer, intent(in) :: topos
end subroutine

attributes(intrinsic,std_gnu)&
elemental subroutine imvbits(from,frompos,len,to,topos)
  integer, intent(in) :: from
  integer, intent(in) :: frompos
  integer, intent(in) :: len
  integer, intent(inout) :: to
  integer, intent(in) :: topos
end subroutine

attributes(intrinsic,std_gnu)&
elemental subroutine bmvbits(from,frompos,len,to,topos)
  integer, intent(in) :: from
  integer, intent(in) :: frompos
  integer, intent(in) :: len
  integer, intent(inout) :: to
  integer, intent(in) :: topos
end subroutine

attributes(intrinsic,impure,std_gnu)&
function fe_runtime_error(msg)
  character, intent(in) :: msg
  !
  type(*), dimension(..) :: fe_runtime_error
end function

attributes(intrinsic,impure,std_gnu)&
subroutine alarm(seconds,handler,status)
  integer, intent(in) :: seconds
  type(*), dimension(..), intent(in) :: handler
  integer, intent(out), optional :: status
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine srand(seed)
  integer(4), intent(in) :: seed
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine exit(status)
  integer, intent(in), optional :: status
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine fgetc(unit,c,status)
  integer, intent(in) :: unit
  character, intent(out) :: c
  integer, intent(out), optional :: status
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine fget(c,status)
  character, intent(out) :: c
  integer, intent(out), optional :: status
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine flush(unit)
  integer, intent(in), optional :: unit
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine fputc(unit,c,status)
  integer, intent(in) :: unit
  character, intent(in) :: c
  integer, intent(out), optional :: status
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine fput(c,status)
  character, intent(in) :: c
  integer, intent(out), optional :: status
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine free(ptr)
  integer, intent(inout) :: ptr
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine fseek(unit,offset,whence,status)
  integer, intent(in) :: unit
  integer, intent(in) :: offset
  integer, intent(in) :: whence
  integer, intent(out), optional :: status
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine ftell(unit,offset)
  integer, intent(in) :: unit
  integer, intent(out) :: offset
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine hostnm(count,status)
  character, intent(out) :: count
  integer, intent(out), optional :: status
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine kill(pid,sig,status)
  integer, intent(in) :: pid
  integer, intent(in) :: sig
  integer, intent(out), optional :: status
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine link(path1,path2,status)
  character, intent(in) :: path1
  character, intent(in) :: path2
  integer, intent(out), optional :: status
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine perror(string)
  character, intent(in) :: string
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine rename(path1,path2,status)
  character, intent(in) :: path1
  character, intent(in) :: path2
  integer, intent(out), optional :: status
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine sleep(seconds)
  integer, intent(in) :: seconds
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine fstat(unit,values,status)
  integer, intent(in) :: unit
  integer, intent(out) :: values
  integer, intent(out), optional :: status
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine lstat(name,values,status)
  character, intent(in) :: name
  integer, intent(out) :: values
  integer, intent(out), optional :: status
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine stat(name,values,status)
  character, intent(in) :: name
  integer, intent(out) :: values
  integer, intent(out), optional :: status
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine signal(number,handler,status)
  integer, intent(in) :: number
  type(*), dimension(..), intent(in) :: handler
  integer, intent(out), optional :: status
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine symlnk(path1,path2,status)
  character, intent(in) :: path1
  character, intent(in) :: path2
  integer, intent(out), optional :: status
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine system(command,status)
  character, intent(in) :: command
  integer, intent(out), optional :: status
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine ttynam(unit,name)
  integer, intent(in) :: unit
  character, intent(out) :: name
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine umask(mask,old)
  integer, intent(in) :: mask
  integer, intent(out), optional :: old
end subroutine

attributes(intrinsic,impure,std_gnu)&
subroutine unlink(path,status)
  character, intent(in) :: path
  integer, intent(out), optional :: status
end subroutine

