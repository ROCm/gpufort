! See also "lib-16-2.f90".
! { dg-do run }
! { dg-skip-if "" { *-*-* } { "*" } { "-DACC_MEM_SHARED=0" } }

program main
    use openacc
    implicit none
  
    integer, parameter :: N = 1000
    integer(4) :: x(N)
    integer :: i
    ! integer :: async = 5

    call acc_set_default_async(5)
    
    do i = 1, N
      x(i) = 3
    end do 
  
    call acc_copyin (x)
  
    do i = 1, N
      x(i) = i + 1
    end do 
  
    call acc_update_device_async (x, 4*N, acc_async_noval)
    
    ! We must wait for the update to be done.
    call acc_wait (acc_async_noval)
    
    call acc_copyout_async (x, 4*N, acc_async_noval)
  
    call acc_wait (acc_async_noval)
  
    do i = 1, N
      if (x(i) /= i + 1) stop 2
    end do 
  
    call acc_copyin (x, 4*N)
    
    call acc_update_self_async (x, 4*N, acc_async_noval)
      
    call acc_wait (acc_async_noval)
  
    do i = 1, N
      if (x(i) /= i + 1) stop 4
    end do 
  
    call acc_delete_async (x, acc_async_noval)
  
    call acc_wait (acc_async_noval)

    print *, "PASSED"
      
  end program
  