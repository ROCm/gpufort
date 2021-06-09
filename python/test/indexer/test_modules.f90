module mymod1
  integer :: a
  integer, parameter :: n = 100
  real :: c(n,n)
 
  !$acc declare create(c)
  attributes(device) :: c

  type mytype
    real :: b(n)
  end type 
end module mymod1

module mymod2
  integer :: a
  integer, parameter :: n = 100
  real :: c(n,n)
 
  !$acc declare create(c)
  attributes(device) :: c

  type :: mytype
    real :: b(n)
  end type 

contains
  subroutine func(a)
    integer,intent(in) :: a
  end subroutine
  
  function func2(a)
    integer,intent(in) :: a
    integer :: func2
    func2 = a 
  end function
  
end module mymod2
