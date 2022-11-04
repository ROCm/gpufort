program main
    use openacc
    call acc_init(acc_device_default)
    call acc_shutdown(acc_device_default)
end program