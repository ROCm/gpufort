program main
    use openacc
    call acc_set_device_type(acc_device_default)
    print *, acc_get_device_type()
end program