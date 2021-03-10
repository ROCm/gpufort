program main
   ! begin of program

#ifdef __HIP
   use vector_add_kernels
   use iso_c_binding
   use openacc_gomp
   implicit none
#else
   implicit none
#endif
   integer, parameter :: N = 1000
   integer :: i
   integer(4) :: x(N), y(N), y_exact(N)

   do i = 1, N
      y_exact(i) = 3
   end do

#ifdef __HIP
   call goacc_data_start(acc_device_default, mappings=[goacc_map_copy(x(1:N)), goacc_map_copy(y(1:N))])
#else
   !$acc data copy(x(1:N),y(1:N))
#endif

#ifdef __HIP
   call goacc_data_start(acc_device_default)
   ! extracted to HIP C++ file
   ! TODO(gpufort) fix arguments
   CALL launch_krnl_cecba2_14_auto(0, c_null_ptr, N, dev_x, size(x, 1), lbound(x, 1), dev_y, size(y, 1), lbound(y, 1))
   call acc_wait_all()
   call goacc_data_end()
#else
   !$acc parallel loop
   do i = 1, N
      x(i) = 1
      y(i) = 2
   end do
#endif

#ifdef __HIP
   call goacc_data_start(acc_device_default)
   ! extracted to HIP C++ file
   ! TODO(gpufort) fix arguments
   CALL launch_krnl_3207b9_20_auto(0, c_null_ptr, N, dev_y, size(y, 1), lbound(y, 1), dev_x, size(x, 1), lbound(x, 1))
   call acc_wait_all()
   call goacc_data_end()
   call goacc_data_end()
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
