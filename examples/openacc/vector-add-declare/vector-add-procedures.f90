program main
  ! begin of program
      
  implicit none
  integer, parameter :: N = 1000
  integer :: i
  integer(4) :: x(N), y(N), y_exact(N)

  do i = 1, N
    y_exact(i) = 3
  end do

  do i = 1, N
    x(i) = 1
    y(i) = 2
  end do
  
  call vector_add_gpu(x,y,.FALSE.)
  call vector_add_gpu(x,y,.TRUE.) ! return prematurely; do not perform addition
 
  do i = 1, N
    if ( y_exact(i) .ne.&
            y(i) ) ERROR STOP "GPU and CPU result do not match"
  end do

  print *, "PASSED"

contains

subroutine vector_add_gpu(x,y,return_prematurely)
  integer(4) :: x(:), y(:)
  logical    :: return_prematurely
  !$acc declare copyin(x,y)
  
  ! extra return statement
  if ( return_prematurely ) return
  
  !$acc parallel loop
  do i = 1, N
    y(i) = x(i) + y(i)
  end do
  
  !$acc update host(y)
end subroutine

end program
