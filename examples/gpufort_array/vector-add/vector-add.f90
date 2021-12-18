! SPDX-License-Identifier: MIT
! Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
program main
  use hipfort
  use hipfort_check
  use gpufort_array
  implicit none
  integer, parameter :: N = 40000
  real :: x(N), y(N), a
  type(gpufort_array1) :: x_d, y_d
  type(dim3)  :: grid, tBlock
  type(c_ptr) :: stream = c_null_ptr
  !
  tBlock = dim3(256,1,1)
  grid   = dim3(ceiling(real(N)/tBlock%x),1,1)

  call hipCheck(gpufort_array_init(x_d,x))
  call hipCheck(gpufort_array_init(y_d,y))

  x = 1.0; y = 2.0; a = 2.0
  call hipCheck(gpufort_array_copy_to_device(x_d))
  call hipCheck(gpufort_array_copy_to_device(y_d))

  call launch_vecadd_kernel(grid,tBlock,0,stream,y_d,a,x_d)
  call hipCheck(hipStreamSynchronize(stream))

  call hipCheck(gpufort_array_copy_to_host(y_d))
  
  call hipCheck(gpufort_array_destroy(x_d))
  call hipCheck(gpufort_array_destroy(y_d))

  write(*,*) 'Max error: ', maxval(abs(y-4.0))
end program main