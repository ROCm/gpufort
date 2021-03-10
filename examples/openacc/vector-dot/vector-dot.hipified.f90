program main
   use iso_c_binding
   use gpufort_acc_runtime
   use vector_dot_kernels
   implicit none

   integer, parameter :: N = 1000
   integer :: i
   integer(4) :: x(N), y(N), res

   type(c_ptr) :: y_d
   type(c_ptr) :: x_d

   call gpufort_acc_enter_region()
   x_d = gpufort_acc_copy(x(1:n))
   y_d = gpufort_acc_copy(y(1:n))

   call gpufort_acc_enter_region()
   ! extracted to HIP C++ file
   ! TODO(gpufort) fix arguments
   CALL launch_krnl_cecba2_8_auto(0, c_null_ptr, N, x_d, size(x, 1), lbound(x, 1), y_d, size(y, 1), lbound(y, 1))
   call gpufort_acc_exit_region()

   call gpufort_acc_wait()

   res = 0
   call gpufort_acc_enter_region()
   ! extracted to HIP C++ file
   ! TODO(gpufort) fix arguments
   CALL launch_krnl_e7eb26_15_auto(0, c_null_ptr, N, res, x_d, size(x, 1), lbound(x, 1), y_d, size(y, 1), lbound(y, 1))
   call gpufort_acc_exit_region()

   call gpufort_acc_wait()

   call gpufort_acc_exit_region()

   if (res .ne. N*2) ERROR STOP "FAILED"
   PRINT *, "PASSED"

end program
