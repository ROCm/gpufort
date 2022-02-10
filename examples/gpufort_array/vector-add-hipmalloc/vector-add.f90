! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
program main
  use hipfort
  use hipfort_check
  use gpufort_array
  implicit none
  integer, parameter :: N = 40000
  real :: x(N), y(N), a
  real,pointer :: x_d(:), y_d(:)
  type(dim3)  :: grid, tBlock
  type(c_ptr) :: stream = c_null_ptr
  !
  tBlock = dim3(256,1,1)
  grid   = dim3(ceiling(real(N)/tBlock%x),1,1)

  x = 1.0; y = 2.0; a = 2.0

  call hipCheck(hipMalloc(x_d,source=x))
  call hipCheck(hipMalloc(y_d,source=y))

  call launch_vecadd_kernel(grid,tBlock,0,stream,&
    gpufort_array_wrap_device_ptr(y_d,lbound(y)),a,&
    gpufort_array_wrap_device_ptr(x_d))
  call hipCheck(hipStreamSynchronize(stream))

  call hipCheck(hipMemcpy(y,y_d,hipMemcpyDeviceToHost))
  call hipCheck(hipFree(x_d))
  call hipCheck(hipFree(y_d))

  write(*,*) 'Max error: ', maxval(abs(y-4.0))
end program main