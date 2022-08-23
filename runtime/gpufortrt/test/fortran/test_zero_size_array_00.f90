program main
  use iso_c_binding
  integer,parameter :: n = 0
  logical(c_bool) :: a(n) ! legal            
  logical(c_bool),allocatable :: b(:)
  logical(c_bool),pointer :: c(:)

  allocate(b(n),c(n)) ! legal
  !$acc data copyin(a,b,c)

  !$acc update device(a)
  !$acc update device(b)
  !$acc update device(c)
  
  !$acc update host(a)
  !$acc update host(b)
  !$acc update host(c)

  !$acc end data
  deallocate(b,c)
end program
