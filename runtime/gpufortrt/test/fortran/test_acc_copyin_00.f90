program main
    ! begin of program
    use openacc
    implicit none
    integer, parameter :: N = 100
    integer :: i
    integer :: x(N), y(N)
    logical:: xPresent = .FALSE., yPresent = .FALSE. 
  
    call acc_init(acc_device_default)
    do i = 1, N
      y(i) = 3
      x(i) = 2
    end do
    call acc_copyin(x, N)
    xPresent = acc_is_present(x,N)
    yPresent = acc_is_present(y,N)
    if ( xPresent .AND. .not. yPresent) then 
        print *, "FINE!"
    else
        ERROR STOP "Arrays is not present"
    end if
    call acc_shutdown(acc_device_default)
  end program
  