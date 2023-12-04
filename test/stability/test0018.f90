! demonstrates: hiding and unused ambiguous references
module mod1
  integer, parameter :: c = 10
end module

module mod2
  integer, parameter :: c = 200
end module

program main
  use mod1
  use mod2
  implicit none
  integer, parameter :: a = 1
contains
  subroutine foo(arg)
    integer, parameter :: a = 2
  end subroutine
end program main
