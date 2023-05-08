program main
  ! begin of program
      
  implicit none
  integer, parameter :: N = 100
  integer :: i
  integer(4) :: x(N), y(N), y_exact(N)

  do i = 1, N
    y_exact(i) = 3
  end do

  !$acc serial copy(x,y)
  x = 1
  y(1:N) = x(1:n) + x

  do i = 1, N
    y(i) = x(i) + y(i)
  end do
  !$acc end serial
  
  do i = 1, N
    if ( y_exact(i) .ne.&
            y(i) ) ERROR STOP "GPU and CPU result do not match"
  end do

  print *, "PASSED"

end program
