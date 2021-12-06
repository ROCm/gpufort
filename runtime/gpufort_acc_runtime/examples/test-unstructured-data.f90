! SPDX-License-Identifier: MIT                                                
! Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
module my_module
  contains
        
  subroutine init()
     use gpufort_acc_runtime
     implicit none
     print *, "initializing"
     !$acc init
     call gpufort_acc_init()
  end subroutine
  
  subroutine shutdown()
     use gpufort_acc_runtime
     implicit none
     print *, "shutting down"
     !$acc shutdown
     call gpufort_acc_shutdown()
  end subroutine
  
  subroutine my_subroutine(a,b)
     use iso_c_binding
     use gpufort_acc_runtime
     implicit none
     real(8),allocatable,target :: a(:,:), b(:,:)
     type(c_ptr) :: a_d, b_d

     print *, "entering subregion"
     !$acc kernels present(a) copyin(b(:,:))
     call gpufort_acc_enter_region()
     a_d = gpufort_acc_present(a)
     b_d = gpufort_acc_copyin(b(:,:))
     call gpufort_acc_runtime_print_summary() 
     !$acc update host(b)
     call gpufort_acc_update_host(b)
     !$acc update device(b)
     call gpufort_acc_update_device(b)
     !$acc end kernels
     call gpufort_acc_exit_region()
     print *, "exiting subregion" 
  end subroutine
end module

program test_acc
  use iso_c_binding
  use gpufort_acc_runtime
  use my_module
  !
  real(8),allocatable,target :: a(:,:), b(:,:)
  type(c_ptr) :: a_d, b_d
  integer :: i
  !
  allocate(a(10,10))
  allocate(b(5,5))

  call init()

  do i = 1, 3
    print *, "entering region"
    
    !$acc enter data copyin(a(:,:))
    call gpufort_acc_enter_region()
    a_d = gpufort_acc_copyin(a(:,:))
    call gpufort_acc_runtime_print_summary() 

    if ( i .eq. 2 ) then
       call my_subroutine(a,b)
    endif
    !$acc update host(a(:,:))
    call gpufort_acc_update_host(a(:,:))
    !$acc update device(a)
    call gpufort_acc_update_device(a)
    !$acc exit data
    call gpufort_acc_exit_region()
   
    ! b_d = gpufort_acc_present(b) ! This will fail correctly
   
    print *, "exiting region" 
  end do

  call gpufort_acc_runtime_print_summary() 
  
  ! a_d = gpufort_acc_present(a) ! This will fail correctly
 
  call shutdown() 

end program 
