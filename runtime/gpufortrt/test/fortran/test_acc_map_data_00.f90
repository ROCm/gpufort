program main
    use iso_c_binding
    implicit none
    type (c_ptr), target :: cptr
    integer, parameter :: N = 5
    integer, pointer :: fptr(:)
    integer, target :: data_arg(N)
    integer :: i = 0

    interface
    function acc_malloc (num_bytes) result(res) &
      bind (C)
      use iso_c_binding
      integer (c_size_t), value :: num_bytes
      !
      type(c_ptr)  :: res
    end function
    end interface

    interface
    subroutine acc_map_data (data_arg, data_dev, num_bytes) &
      bind (C)
      use iso_c_binding
      type (c_ptr), value:: data_arg
      type (c_ptr), value:: data_dev
      integer (c_size_t), value :: num_bytes
    end subroutine
    end interface

    cptr = acc_malloc (N * sizeof (fptr(N)))
    call c_f_pointer (cptr, fptr, [N])

    call initialize_device_memory (fptr, N)

    call acc_map_data (c_loc(data_arg), cptr, N * sizeof (fptr(N)))
    
    do i = 1, N
        if (data_arg(i) .ne. i * 2) ERROR STOP "Results do not match"
    end do
      
    print *, "PASSED"

    contains
    subroutine initialize_device_memory (fptr, N)
      use openacc
      implicit none
      integer :: N
      integer :: fptr(N)
      integer :: i = 0
      
      !$acc data deviceptr (fptr)

      !$acc parallel 
        do i = 1, N
          fptr(i) = i * 2
        end do
      !$acc end parallel

    end subroutine
  
  end program main  