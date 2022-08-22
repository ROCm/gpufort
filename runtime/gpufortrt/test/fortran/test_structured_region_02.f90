program main
  use iso_c_binding
  integer,parameter :: n = 10
  logical(c_bool) :: x(n),y(2*n)
  !$acc init
  !$acc data copy(x,y(1:n),y(n+1:2*n))
  !$acc end data
  !$acc shutdown
end program