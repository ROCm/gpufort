! SPDX-License-Identifier: MIT                                                
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
module my_module
  contains
        
  subroutine init()
     use gpufortrt
     implicit none
     print *, "initializing"
     !$acc init
     call gpufortrt_init()
  end subroutine
  
  subroutine shutdown()
     use gpufortrt
     implicit none
     print *, "shutting down"
     !$acc shutdown
     call gpufortrt_shutdown()
  end subroutine
  
  subroutine my_subroutine(a,b)
     use iso_c_binding
     use gpufortrt
     implicit none
     real(8),allocatable,target :: a(:,:), b(:,:)
     type(c_ptr) :: a_d, b_d

     print *, "entering subregion"
     !$acc kernels present(a) copyin(b(:,:))
     call gpufortrt_enter_region()
     a_d = gpufortrt_present(a)
     b_d = gpufortrt_copyin(b(:,:))
     call gpufortrt_print_summary() 
     !$acc update host(b)
     call gpufortrt_update_host(b)
     !$acc update device(b)
     call gpufortrt_update_device(b)
     !$acc end kernels
     call gpufortrt_exit_region()
     print *, "exiting subregion" 
  end subroutine
end module

program test_acc
  use iso_c_binding
  use gpufortrt
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
    call gpufortrt_enter_region()
    a_d = gpufortrt_copyin(a(:,:))
    call gpufortrt_print_summary() 

    if ( i .eq. 2 ) then
       call my_subroutine(a,b)
    endif
    !$acc update host(a(:,:))
    call gpufortrt_update_host(a(:,:))
    !$acc update device(a)
    call gpufortrt_update_device(a)
    !$acc exit data
    call gpufortrt_exit_region()
   
    ! b_d = gpufortrt_present(b) ! This will fail correctly
   
    print *, "exiting region" 
  end do

  call gpufortrt_print_summary() 
  
  ! a_d = gpufortrt_present(a) ! This will fail correctly
 
  call shutdown() 

end program 