program main
  integer,parameter :: n = 10
  logical(c_bool) :: x(n), y(n)
  !$acc init
  !$acc data copy(x)
    !$acc data present(x) copyout(y)
    !$acc end data
  !$acc end data
  !$acc shutdown
end program