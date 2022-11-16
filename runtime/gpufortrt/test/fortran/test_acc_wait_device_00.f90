program main
    use openacc
    implicit none
    integer :: n = 0
    call acc_wait_device(n , 0)
end program