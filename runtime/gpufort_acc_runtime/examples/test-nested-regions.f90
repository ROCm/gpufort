! SPDX-License-Identifier: MIT                                                
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
program test_acc
  use iso_c_binding
  use gpufort_acc_runtime
  !
  real(8),allocatable,target :: a(:,:), b(:,:), c(:), d(:), e(:)
  type(c_ptr) :: a_d, b_d, c_d, d_d, e_d
  integer :: i
  !
  allocate(a(10,10))
  allocate(b(5,5))
  allocate(c(20))
  allocate(d(120))
  allocate(e(122))

  print *, "initializing"
  !$acc init
  call gpufort_acc_init()

  do i = 1, 4
    print *, "entering region"
    call gpufort_acc_enter_region()
    a_d = gpufort_acc_copyin(a(:,:))
    d_d = gpufort_acc_copy(d(:))
    e_d = gpufort_acc_copyout(e(:))
    !call gpufort_acc_runtime_print_summary() 

    if ( mod(i,2) .eq. 0 ) then
      print *, "entering subregion"
      call gpufort_acc_enter_region()
      a_d = gpufort_acc_present(a(:,5))
      b_d = gpufort_acc_copyin(b(:,:))
      c_d = gpufort_acc_create(c(:))
      !call gpufort_acc_runtime_print_summary() 
      call gpufort_acc_update_host(b)
      call gpufort_acc_update_device(b)
      call gpufort_acc_exit_region()
      print *, "exiting subregion" 
    endif
    call gpufort_acc_update_host(a)
    call gpufort_acc_update_device(a)
    call gpufort_acc_exit_region()
    print *, "exiting region" 
  end do

  call gpufort_acc_runtime_print_summary() 
  
  !a_d = gpufort_acc_present(c_loc(a)) ! This will fail correctly
  
  print *, "shutting down"
  !$acc shutdown
  call gpufort_acc_shutdown()

end program 