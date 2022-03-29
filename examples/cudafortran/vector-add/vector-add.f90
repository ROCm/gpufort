! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
program main
  use cudafor
  implicit none
  integer, parameter :: N = 400
  real :: x(N), y(N), a
  real, device :: x_d(N)
  real, allocatable :: y_d(:)
  type(dim3) :: grid, tBlock
  integer :: i
  !@cuf integer :: ierr

  attributes(device) :: y_d
  allocate(y_d(N))

  tBlock = dim3(256,1,1)
  grid = dim3(ceiling(real(N)/tBlock%x),1,1)

  x = 1.0; y = 2.0; a = 2.0
  x_d = x
  y_d = y

#define xi x_d(i)

  !$cuf kernel do(1) <<<grid, tBlock>>>
  do i=1,size(y_d,1)
    y_d(i) = y_d(i) + a*xi
  end do
  !@cuf ierr = cudaDeviceSynchronize()
  !@cuf if ( ierr .ne. 0 ) ERROR STOP "kernel launch failed!"

  y = y_d

  deallocate(y_d)

#define max_err(a) maxval(abs(a-4.0))

  write(*,*) 'Max error: ', max_err(y) 
end program main
