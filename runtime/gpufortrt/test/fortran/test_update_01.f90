program main
  use iso_c_binding
  integer,parameter :: n = 10
  logical(c_bool) :: a(1:2*n)

  ! map full array
  !$acc data create(a)

  !$acc update device(a(1:n))
  !$acc update device(a(n+1:2*n))
  !$acc update host(a(1:n))
  !$acc update host(a(n+1:2*n))

  !$acc end data
  
end program
