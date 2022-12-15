program main
    use openacc
    use iso_c_binding
    implicit none
    !
    integer :: dev_type = acc_device_default
    integer :: dev_num
    integer :: dev
    integer(c_size_t) :: val
    dev_num = acc_get_num_devices(dev_type)
  
    do dev = 0, dev_num
       print *, "Device ", dev
  
       val = acc_get_property (dev, dev_type, acc_property_memory)
       print *, "    Total memory: ", val
       if (val < 0) then
          print *, "acc_property_memory should not be negative."
          stop 1
       end if
    end do
end program
  