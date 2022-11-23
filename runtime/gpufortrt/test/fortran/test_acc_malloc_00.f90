program main
    use openacc
    use iso_c_binding
    implicit none
    type (c_ptr) :: cptr
    integer, parameter :: N = 1000
    integer(4), pointer :: fptr(:)
    logical:: fptrPresent = .FALSE.
  
    interface
        type (c_ptr) function acc_malloc (num_bytes) &
            bind (C)
            use iso_c_binding
            integer (c_size_t), value :: num_bytes
        end function
    end interface
  
    cptr = acc_malloc (N * sizeof (fptr(N)))
    call c_f_pointer (cptr, fptr, [N])

    fptrPresent = acc_is_present(fptr)
    if ( fptrPresent ) then 
        print *, "FINE!"
    else
        ERROR STOP "Arrays is not present"
    end if
  
end program main