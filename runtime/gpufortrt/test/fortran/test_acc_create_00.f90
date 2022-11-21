program main
    ! begin of program
    use openacc
    implicit none
    integer, parameter :: N = 1000
    integer :: i
    integer(4) :: x(N), y(N), y_exact(N)

    do i = 1, N
      y_exact(i) = 3
    end do
  
    call acc_create(x,N*4)
    call acc_create(y,N*4)

    !$acc parallel loop present(x,y)
    !
    do i = 1, N
      x(i) = 1
      y(i) = 2
    end do
    
    !$acc parallel loop
    do i = 1, N
      y(i) = x(i) + y(i)
    end do
  
    call acc_copyout(y,N*4)
    
    do i = 1, N
      if ( y_exact(i) .ne.&
              y(i) ) ERROR STOP "GPU and CPU result do not match"
    end do
  
    print *, "PASSED"
  
  end program
  