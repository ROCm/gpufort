module myothermod
  private
  integer, parameter :: prec = selected_real_kind(5)
  integer, save, public :: other = 1
  public :: prec
end module

module mymod
  use myothermod, only: prec, other

contains

subroutine mysub()
  integer :: a
  a = 1
end subroutine

end module
