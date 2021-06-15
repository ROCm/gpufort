module base
  integer :: z1, z2
end module


module simple
  use base, only: z1 => abc1
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

module nested_subprograms
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
      real,intent(in) :: a ! scoper: should hide the a in scoper
      integer :: func3
      integer :: e(n,n) ! scoper: should hide the e in module
      func3 = a 
    end function
  end function
  
end module nested_subprograms
