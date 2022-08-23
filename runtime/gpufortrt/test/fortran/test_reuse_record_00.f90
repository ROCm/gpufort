program main
  use iso_c_binding
  integer,parameter :: n = 10
  logical(c_bool) :: x(n)

  !$acc init

  do i = 1,3
  !$acc data copy(x)
  !$acc end data
  end do
  
  !$acc shutdown
end program
