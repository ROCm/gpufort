program main
  ! begin of program
      
  implicit none
  integer, parameter :: N = 1000
  integer :: i
  integer(4) :: y(N), y_exact(N)
  type struct_t
    integer(4) :: coeff
    integer(4),allocatable :: x(:)
    integer(4) :: N
  end type
  type(struct_t) :: struct

  struct%coeff = 1
  struct%N = N
  allocate(struct%x(N))

  do i = 1, N
    y_exact(i) = 3
  end do

  !$acc data copy(struct%x(1:N),y(1:N)) copyin(struct)
 
  !$acc parallel loop present(y(1:N))
  do i = 1, N
    y(i) = 2
  end do


  !$acc update device(struct%n) ! ignore
 
  !$acc kernels
  struct%x(1:struct%N) = 1
  !$acc end kernels

  !$acc parallel loop
  do i = 1, N
    y(i) = struct%x(i) + struct%coeff*y(i)
  end do
  
  !$acc update host(struct%x) if_present ! translate
  
  !$acc end data
  
  do i = 1, N
    if ( y_exact(i) .ne.&
            y(i) ) ERROR STOP "GPU and CPU result do not match"
  end do

  deallocate(struct%x)

  print *, "PASSED"

end program
