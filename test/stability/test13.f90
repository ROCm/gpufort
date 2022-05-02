module mymod
  integer, parameter :: prec = selected_real_kind(5)

contains

subroutine mysub()
  integer :: a
  a = 1
end subroutine

end module
