program main
    use openacc
    integer(acc_device_kind) :: dev_type
    dev_type = acc_device_default
    print *, acc_get_device_num(dev_type)
end program