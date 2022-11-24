program main
  use iso_c_binding
  implicit none
  type (c_ptr) :: cptr
  integer, parameter :: N = 1000
  integer, pointer :: fptr(:)
  integer :: test_array(N)
  integer :: i = 0

  interface
    type (c_ptr) function acc_malloc (num_bytes) &
      bind (C)
      use iso_c_binding
      integer (c_size_t), value :: num_bytes
    end function
  end interface

  cptr = acc_malloc (N * sizeof (fptr(N)))
  call c_f_pointer (cptr, fptr, [N])
    
  call dummy_subroutine (fptr, test_array, N)
    
  do i = 1, N
      if (test_array(i) .ne. i) ERROR STOP "Results do not match"
  end do
    
  print *, "PASSED"

  contains
  subroutine dummy_subroutine (fptr, test_array, N)
      use openacc
      implicit none
      integer :: N
      integer :: test_array(N)
      integer :: fptr(N)
      integer :: i = 0
      
      !$acc data deviceptr (fptr)
  
      call acc_copyin(test_array)
      !$acc parallel 
        do i = 1, N
          fptr(i) = i
          test_array(i) = fptr(i)
        end do
      !$acc end parallel
      call acc_copyout(test_array)
      
    end subroutine

end program main  