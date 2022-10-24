function foo(bar) result(retval)
  implicit none
  !$acc routine seq
  integer :: bar
  integer :: retval
end function
