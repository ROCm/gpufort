module myothermod
  private
  integer, parameter :: prec = selected_real_kind(5)
  public :: prec
end module

module mymod
  use myothermod, only: prec

contains

subroutine mysub()
  integer :: a
  a = 1
end subroutine

end module
