! SPDX-License-Identifier: MIT                                                
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
program test_acc
  use iso_c_binding
  use gpufortrt
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
  call gpufortrt_init()

  do i = 1, 4
    print *, "entering region"
    call gpufortrt_enter_region()
    a_d = gpufortrt_copyin(a(:,:))
    d_d = gpufortrt_copy(d(:))
    e_d = gpufortrt_copyout(e(:))
    !call gpufortrt_print_summary() 

    if ( mod(i,2) .eq. 0 ) then
      print *, "entering subregion"
      call gpufortrt_enter_region()
      a_d = gpufortrt_present(a(:,5))
      b_d = gpufortrt_copyin(b(:,:))
      c_d = gpufortrt_create(c(:))
      !call gpufortrt_print_summary() 
      call gpufortrt_update_host(b)
      call gpufortrt_update_device(b)
      call gpufortrt_exit_region()
      print *, "exiting subregion" 
    endif
    call gpufortrt_update_host(a)
    call gpufortrt_update_device(a)
    call gpufortrt_exit_region()
    print *, "exiting region" 
  end do

  call gpufortrt_print_summary() 
  
  !a_d = gpufortrt_present(c_loc(a)) ! This will fail correctly
  
  print *, "shutting down"
  !$acc shutdown
  call gpufortrt_shutdown()

end program 