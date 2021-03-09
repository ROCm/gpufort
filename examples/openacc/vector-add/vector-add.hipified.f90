program main
   ! begin of program

   use vector_add_kernels
   use iso_c_binding
   use openacc_gomp
   implicit none

   integer, parameter :: N = 1000
   integer :: i
   integer(4) :: x(N), y(N), y_exact(N)

   do i = 1, N
      y_exact(i) = 3
   end do

   call goacc_data_start(acc_device_default, mappings=[goacc_map_copy(x(1:N)), goacc_map_copy(y(1:N))])

   call goacc_data_start(acc_device_default)
   ! extracted to HIP C++ file
   ! TODO(gpufort) fix arguments
   CALL launch_krnl_cecba2_14_auto(0, c_null_ptr, N, x_d, size(x, 1), lbound(x, 1), y_d, size(y, 1), lbound(y, 1))
   call acc_wait_all()
   call goacc_data_end()

   call goacc_data_start(acc_device_default)
   ! extracted to HIP C++ file
   ! TODO(gpufort) fix arguments
   CALL launch_krnl_3207b9_20_auto(0, c_null_ptr, N, y_d, size(y, 1), lbound(y, 1), x_d, size(x, 1), lbound(x, 1))
   call acc_wait_all()
   call goacc_data_end()
   call goacc_data_end()

   do i = 1, N
      if (y_exact(i) .ne. &
          y(i)) ERROR STOP "GPU and CPU result do not match"
   end do

   print *, "PASSED"

end program
