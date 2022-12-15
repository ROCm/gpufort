! { dg-do run }

program main
    use openacc
    implicit none
    integer :: i
    integer, parameter :: N = 1000
    integer(4) :: x(N)
      
    call acc_set_default_async(1)
    call acc_copyin_async(x,N*4,acc_async_noval)

    !$acc parallel  
      do i = 1, N
        x(i) = 1
      end do
    !$acc end parallel
  
    call acc_wait_async (0, acc_async_noval)
  
    ! Test unseen async-argument.
    if (acc_async_test (12) .neqv. .TRUE.) stop 1
    call acc_wait_async (12, acc_async_noval)
  
    call acc_wait (1)
  
    if (acc_async_test (0) .neqv. .TRUE.) stop 2
    if (acc_async_test (1) .neqv. .TRUE.) stop 3
    if (acc_async_test (2) .neqv. .TRUE.) stop 4
  
    print *, "acc_get_default_async: ", acc_get_default_async()
    
  end program
  