program main
   ! begin of program

   implicit none
   integer, parameter :: N = 1000
   integer :: i
   integer(4) :: x(N), y(N), y_exact(N)

   do i = 1, N
      y_exact(i) = 3
   end do

#ifdef __GPUFORT
   !$omp target data map(tofrom:x(1:N),y(1:N))
#else
   !$acc data copy(x(1:N),y(1:N))
#endif

#ifdef __GPUFORT
   !$omp target teams distribute parallel do
   do i = 1, N
      x(i) = 1
      y(i) = 2
   end do
#else
   !$acc parallel loop
   do i = 1, N
      x(i) = 1
      y(i) = 2
   end do
#endif

#ifdef __GPUFORT
   !$omp target teams distribute parallel do
   do i = 1, N
      y(i) = x(i) + y(i)
   end do
   !$omp end target data
#else
   !$acc parallel loop
   do i = 1, N
      y(i) = x(i) + y(i)
   end do
   !$acc end data
#endif

   do i = 1, N
      if (y_exact(i) .ne. &
          y(i)) ERROR STOP "GPU and CPU result do not match"
   end do

   print *, "PASSED"

end program
