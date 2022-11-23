program main
  ! begin of program
  use openacc
  implicit none
  integer, parameter :: N = 1000
  integer :: i
  integer(4) :: x(N), y(N), y_exact(N)
  logical:: xPresent = .FALSE., yPresent = .FALSE. 

  do i = 1, N
    y_exact(i) = 3
  end do

  call acc_copyin(x)
  call acc_copyin(y)

  xPresent = acc_is_present(x)
  yPresent = acc_is_present(y, int(sizeof(y)))
  if ( xPresent .AND. yPresent) then 
      print *, "FINE!"
  else
      ERROR STOP "Arrays is not present"
  end if

  !$acc parallel loop 
  !
  do i = 1, N
    x(i) = 1
    y(i) = 2
  end do
  
  !$acc parallel loop
  do i = 1, N
    y(i) = x(i) + y(i)
  end do

  call acc_copyout(y)
  
  do i = 1, N
    if ( y_exact(i) .ne.&
            y(i) ) ERROR STOP "GPU and CPU result do not match"
  end do

  print *, "PASSED"

end program
