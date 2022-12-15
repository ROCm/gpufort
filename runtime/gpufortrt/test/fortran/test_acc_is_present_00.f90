program main
    ! begin of program
    use openacc
    implicit none
    integer, parameter :: N = 1000
    integer(4) :: x(N), y(N)
    logical:: xPresent = .FALSE., yPresent = .FALSE. 
  
    !$ACC DATA COPY(X(1:N))
    xPresent = acc_is_present(x,N)
    yPresent = acc_is_present(y,N)
    if ( xPresent .AND. .not. yPresent) then 
        print *, "FINE!"
    else
        ERROR STOP "Arrays is not present"
    end if
    
    !$acc end data
  
  end program
  
