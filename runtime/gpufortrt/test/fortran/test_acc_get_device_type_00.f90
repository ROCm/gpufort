program main
    use openacc
    implicit none
    integer(acc_device_kind) :: dev_type
    dev_type = acc_get_device_type()
    print *, dev_type
end program