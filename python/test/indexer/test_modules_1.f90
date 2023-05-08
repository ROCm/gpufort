
! 1. simple_base and base: arrays, scalars, types, device attributes, and use statements
module simple_base
  integer :: z1, z2
end module

module simple
  use simple_base, only: abc1 => z1 ! include z1 but rename it to abc1
  integer :: a
  integer, parameter :: n = 100
  real :: c(n,n)
 
  !$acc declare create(c)
#ifdef CUDA
  attributes(device) :: c
#endif

  type mytype
    real :: b(n)
  end type 
end module simple


! 2. procedure nesting
module nested_procedures
  integer :: a
  integer, parameter :: n = 1000
  real :: e(-n:n,-n:n)

  type,bind(c) :: mytype ! type with same name as in other module
    real*8 :: b(n)
  end type 

contains
  subroutine func(a)
    integer,intent(in) :: a
  end subroutine
  
  function func2(a) result(res)
    integer,intent(in) :: a
    integer :: res
    res = a 
  
  ! Fortran allows single-level of nesting
  contains

    function func3(a)
      real,intent(in) :: a ! gpufort.scope: should hide the a in gpufort.scope
      integer :: func3
      integer :: e(n,n) ! gpufort.scope: should hide the e in module
      func3 = a 
    end function
    
    function func4(a)
      !$acc routine seq
      real,intent(in) :: a ! gpufort.scope: should hide the a in gpufort.scope
      integer :: func3
      integer :: e(n,n) ! gpufort.scope: should hide the e in module
      func3 = a 
    end function
  end function
  
end module nested_procedures


! 3. complex types
module complex_types_base_1
 type type1
   integer,allocatable :: a(:) 
 end type
end module

module complex_types_base_2
  use complex_types_base_1, only: type1
  type type2
    type(type1),allocatable,dimension(:) :: t1list
  end type
end module

module complex_types 
  use complex_types_base_2 ! should also include type1

  type complex_type
    type(type1),pointer,dimension(:)     :: t1list
    type(type2),allocatable,dimension(:) :: t2list
  end type
end module
