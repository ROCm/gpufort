program main
  integer,parameter :: n = 10
  logical :: x(n)
  logical, allocatable :: y(:)
  logical, pointer :: z(:)
  !$acc declare copyin(x) create(y,z) 

  call alloc()
  call dealloc()
contains
  subroutine alloc()
    implicit none
    allocate(y(n),z(n))
  end subroutine

  subroutine dealloc()
    implicit none
    deallocate(y,z)
  end subroutine
end program
