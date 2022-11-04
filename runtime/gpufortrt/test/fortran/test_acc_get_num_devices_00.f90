program main
    use openacc
    implicit none
    integer:: num
    num = acc_get_num_devices(acc_device_default)
    print *, num
end program