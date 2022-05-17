module myothermod
  private
  integer, parameter :: prec = selected_real_kind(5)
  integer, save, public :: other = 1
  integer, parameter, public :: other_param = 2
  public :: prec
end module

module mymod
  use myothermod, only: prec, other2 => other, other_param2 => other_param

contains

subroutine mysub()
  integer :: a
  a = 1
end subroutine

end module
