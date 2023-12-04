! SPDX-License-Identifier: MIT                                                
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
program test_acc
  use openacc_gomp
  use hipfort_check
  use hipfort
  use hipfort_hipblas
  implicit none

  integer :: N = 100, j
  real, allocatable, dimension(:) :: x, y, y_exact
  real, pointer, dimension(:) :: dx, dy

  real, parameter :: alpha = 2.0

  integer, parameter :: bytes_per_element = 4 !float precision

  real :: error
  real, parameter :: error_max = 10*epsilon(error)
  
  type(c_ptr) :: handle = c_null_ptr

  integer :: num_devices, deviceId = -1

  allocate(x(N))
  allocate(y(N))
  allocate(y_exact(N))
 
  write(*,*) "Starting SAXPY test"
    
  num_devices = acc_get_num_devices(acc_device_radeon)
  Print *, "num_devices=", num_devices
  do deviceId = 0, num_devices-1
    Print *, "deviceId=", deviceId
    call acc_set_device_num(deviceId,acc_device_radeon)
    call hipCheck(hipSetDevice(deviceId))
    call hipblasCheck(hipblasCreate(handle))

    do j = 0, N
      x(j) = j
      y(j) = j
    end do

    y_exact(:) = alpha*x(:) + y(:)
  
    !!$acc enter data copy(x(:),y(:,:))
    call GOACC_enter_exit_data(acc_device_default,&
      mappings=[map_copyin(x),map_copyin(y)]&
    )
    
    print *, "acc_is_present(x) = ", acc_is_present(x)
    print *, "acc_is_present(y) = ", acc_is_present(y)
  
    call hipblasCheck(hipblasSaxpy(handle,N,alpha,acc_deviceptr(x),1,acc_deviceptr(y),1))
    call hipCheck(hipDeviceSynchronize())
    
    !!$acc exit data delete(x(:),y(:,:))
    call GOACC_enter_exit_data(acc_device_default,&
      mappings=[map_delete(x),map_copyout(y)]&
    )
    call GOACC_enter_exit_data(acc_device_default)
  
    print *, "acc_is_present(x) = ", acc_is_present(x)
    print *, "acc_is_present(y) = ", acc_is_present(y)
    
    do j = 1,N
      error = abs((y_exact(j) - y(j))/y_exact(j))
        if( error > error_max )then
          write(*,*) "SAXPY FAILED! Error bigger than max! Error = ", error
          ERROR STOP "SAXYP FAILED!"
        end if
    end do

    write(*,*) "SAXPY PASSED!"
    call hipblasCheck(hipblasDestroy(handle))
  end do

  deallocate(x)
  deallocate(y)

  call acc_shutdown(acc_device_radeon)

end program test_acc