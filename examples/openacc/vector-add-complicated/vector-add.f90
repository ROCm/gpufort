! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
module params1
  public
  integer, parameter :: prec = selected_real_kind(15, 307) ! double precision
end module

module params
  use params1
  !private

  integer, parameter, public :: N = 1000,&
                                magnitude = 1
  real(prec), parameter, public :: param = 1.000e-5_prec
end module

program main
  ! begin of program
  use params  
    
  implicit none
  integer :: coeffs(-magnitude:magnitude)
  integer :: i,j,k
  integer(4) :: x(N), y(N,5), y_exact(N)
  logical :: use_gpu = .true.

  integer :: add, mult, a, b
  add(a,b)=a+b ! statement functions
  mult(a,b)=a*b ! statement functions

  do i = 1, N
    y_exact(i) = 3
  end do

  !$acc data copy(x(1:N),y(1:N,3))

  !just an unnecessarily complicated way to fill 1D arrays
  !$acc parallel loop present(x,y(1:N,3)) private(coeffs) & ! inline comment 
  !$acc collapse(3) ! continuation of the above directive
  do 10 k = 1,10 
    do 20 j = 4, -4, -8
      do 20 i = 1, N/2
        coeffs(-1) = 4
        coeffs(-0) = 8
        coeffs(1)  = 2
        
        if ( k .neqv. 1 ) goto 20
        x( ((j-coeffs(-1))/-coeffs(0))*N/coeffs(1)+i ) = 1; y( ((j-coeffs(-1))/-coeffs(0))*N/coeffs(1)+i, 3 ) &

                ! another inline comment
                = 2
  20 continue ! a shared statement number
  10 continue
  
  !$acc parallel loop present(x,y(1:N,3)) if(use_gpu)
  do i = 1, N
    y(i,1) = add(x(i),mult(1,y(i,3)))
  end do
  !$acc end data
  
  do i = 1, N
    if ( y_exact(i) .ne.&
            y(i,3) ) ERROR STOP "GPU and CPU result do not match"
  end do

  print *, "PASSED"

end program
