program main
#ifdef __GPUFORT
   use vector_dot_kernels
   use iso_c_binding
   use gpufort_acc_runtime
   implicit none
#else
   implicit none
#endif
   integer, parameter :: N = 1000
   integer :: i
#ifdef __GPUFORT
   integer(4) :: x(N), y(N), res
   type(c_ptr) :: x_d
   type(c_ptr) :: y_d

#else
   integer(4) :: x(N), y(N), res
#endif

#ifdef __GPUFORT
   call gpufort_acc_enter_region()
   x_d = gpufort_acc_copy(x(1:n))
   y_d = gpufort_acc_copy(y(1:n))
#else
   !$acc data copy(x(1:N),y(1:N))
#endif

#ifdef __GPUFORT
   call gpufort_acc_enter_region()
   ! extracted to HIP C++ file
   ! TODO(gpufort) fix arguments
   CALL launch_krnl_cecba2_8_auto(0, c_null_ptr, N, x_d, size(x, 1), lbound(x, 1), y_d, size(y, 1), lbound(y, 1))
   call gpufort_acc_exit_region()

   call gpufort_acc_wait()
#else
   !$acc parallel loop
   do i = 1, N
      x(i) = 1
      y(i) = 2
   end do
#endif

   res = 0
#ifdef __GPUFORT
   call gpufort_acc_enter_region()
   ! extracted to HIP C++ file
   ! TODO(gpufort) fix arguments
   CALL launch_krnl_e7eb26_15_auto(0, c_null_ptr, N, res, x_d, size(x, 1), lbound(x, 1), y_d, size(y, 1), lbound(y, 1))
   call gpufort_acc_exit_region()

   call gpufort_acc_wait()
#else
   !$acc parallel loop reduction(+:res)
   do i = 1, N
      res = res + x(i)*y(i)
   end do
#endif

#ifdef __GPUFORT
   call gpufort_acc_exit_region()
#else
   !$acc end data
#endif

   if (res .ne. N*2) ERROR STOP "FAILED"
   PRINT *, "PASSED"

end program
