program main
    ! begin of program
    use openacc
    implicit none
    integer, parameter :: N = 100
    integer :: i
    integer(4) :: x(N), y(N), y_exact(N)
    logical:: xPresent = .FALSE., yPresent = .FALSE. 

    do i = 1, N
        y_exact(i) = 3
    end do

    call acc_copyin(x, N)
    call acc_copyin(y, N)

    xPresent = acc_is_present(x,N)
    yPresent = acc_is_present(y,N)

    if ( .not. xPresent .OR. .not. yPresent ) ERROR STOP "Arrays are not present"

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

    call acc_copyout(y,N)

    do i = 1, N
        if ( y_exact(i) .ne.&
                y(i) ) ERROR STOP "GPU and CPU result do not match"
    end do
    
    print *, "PASSED"
  
  end program
  
