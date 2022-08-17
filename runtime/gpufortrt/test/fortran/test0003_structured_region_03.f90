program main
  integer,parameter :: N = 10
  integer :: x(N), y(N)
  !$acc init
  !$acc data copy(x)
  !$acc data present(x) copyout(y)
  !$acc end data
  !$acc end data
  !$acc shutdown
end program
