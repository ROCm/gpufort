program main
  integer,parameter :: N = 10
  integer :: x(N),y(2*N)
  !$acc init
  !$acc data copy(x,y(1:N),y(N+1:2*N))
  !$acc end data
  !$acc shutdown
end program
