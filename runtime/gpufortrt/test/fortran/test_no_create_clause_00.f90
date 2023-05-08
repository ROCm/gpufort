program main  use iso_c_binding
  use iso_c_binding
  integer,parameter :: n = 10
  logical(c_bool) :: x(n),y(n)
  !$acc data copy(x)
  !$acc data no_create(x) no_create(y)

  !$acc end data
  !$acc end data
end program
