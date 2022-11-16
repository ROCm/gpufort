program main
    use openacc
    implicit none
    integer :: dev_num = 1
    call acc_set_device_num(dev_num, acc_device_default)
end program