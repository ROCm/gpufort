program main
  integer,parameter :: n = 10
  logical :: x(n),y(2*n)
  !$acc init
  !$acc data copy(x,y(1:2*n))
  !$acc end data
  !$acc shutdown
end program