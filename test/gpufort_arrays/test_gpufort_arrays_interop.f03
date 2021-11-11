program test_gpufort_arrays_interop
  use gpufort_arrays
  use hipfort_check
  use hipfort
  implicit none
  !
  interface
    subroutine launch_fill_int_array_1(arr) &
        bind(c,name="launch_fill_int_array_1")
      use gpufort_arrays
      type(gpufort_mapped_array_1),intent(inout) :: arr
    end subroutine
    subroutine launch_fill_int_array_2(arr) &
        bind(c,name="launch_fill_int_array_2")
      use gpufort_arrays
      type(gpufort_mapped_array_2),intent(inout) :: arr
    end subroutine
    subroutine launch_fill_int_array_3(arr) &
        bind(c,name="launch_fill_int_array_3")
      use gpufort_arrays
      type(gpufort_mapped_array_3),intent(inout) :: arr
    end subroutine
  end interface
  !
  type(gpufort_mapped_array_1) :: marr1
  type(gpufort_mapped_array_2) :: marr2
  type(gpufort_mapped_array_3) :: marr3
  type(gpufort_mapped_array_4) :: marr4
  type(gpufort_mapped_array_5) :: marr5
  !
  real(c_float),target          :: host_array_3(-1:8,-2:7,-3:6)
  real(c_double_complex),target :: host_array_4(-1:8,-2:7,-3:6,-4:5)
  integer(c_int),dimension(5)                 :: sizes5,lbounds5  
  integer(c_int),pointer,dimension(:,:,:,:,:) :: tmp5, host_array_5
  !
  type(gpufort_mapped_array_1) :: int_marr1
  type(gpufort_mapped_array_2) :: int_marr2
  type(gpufort_mapped_array_3) :: int_marr3
  !
  integer(c_int),target :: int_array_1(-1:8)
  integer(c_int),target :: int_array_2(-1:8,-2:7)
  integer(c_int),target :: int_array_3(-1:8,-2:7,-3:6)
  !
  integer :: i,j,k,n

  ! 
  ! PART 1: Creating mapped arrays
  !

  ! direct access; set from existing pointer
  ! mapped_array,&
  ! bytes_per_element,&
  ! data_host,data_dev,&
  ! sizes, lbounds,&
  ! pinned, copyout_at_destruction) &
  call hipCheck(gpufort_mapped_array_init(marr3,&
    c_float,c_loc(host_array_3),c_null_ptr,&
    shape(host_array_3),lbound(host_array_3),&
    .false._c_bool,.false._c_bool))

  call assert(marr3%data%stride1==1)
  call assert(marr3%data%stride2==10)
  call assert(marr3%data%stride3==100)
  call assert(marr3%data%index_offset==321)
  call assert(marr3%data%num_elements==1000)

  ! overloaded access; set from existing pointer
  call hipCheck(gpufort_mapped_array_init(marr4,&
    host_array_4, lbounds=lbound(host_array_4)))
  
  call assert(marr4%data%stride1==1)
  call assert(marr4%data%stride2==10)
  call assert(marr4%data%stride3==100)
  call assert(marr4%data%stride4==1000)
  call assert(marr4%data%index_offset==4321)
  call assert(marr4%data%num_elements==10000)
  
  ! allocate device AND (pinned) host array
  sizes5   = 4
  lbounds5 = -1
  call hipCheck(gpufort_mapped_array_init(marr5,&
    c_int,sizes5,lbounds=lbounds5,pinned=.true._c_bool))
  call c_f_pointer(marr5%data%data_host,tmp5,shape=sizes5)
  host_array_5(-1:,-1:,-1:,-1:,-1:) => tmp5

  call assert(marr5%data%stride1==1)
  call assert(marr5%data%stride2==4)
  call assert(marr5%data%stride3==16)
  call assert(marr5%data%stride4==64)
  call assert(marr5%data%stride5==256)
  call assert(marr5%data%index_offset==1+4+16+64+256)
  call assert(marr5%data%num_elements==1024)

  ! destroy
  call hipCheck(gpufort_mapped_array_destroy(marr3,c_null_ptr))
  call hipCheck(gpufort_mapped_array_destroy(marr4,c_null_ptr))
  call hipCheck(gpufort_mapped_array_destroy(marr5,c_null_ptr))

  ! 
  ! PART 2: Interfacing
  !
  call hipCheck(gpufort_mapped_array_init(int_marr1,&
    int_array_1, lbounds=lbound(int_array_1)))
  call hipCheck(gpufort_mapped_array_init(int_marr2,&
    int_array_2, lbounds=lbound(int_array_2)))
  call hipCheck(gpufort_mapped_array_init(int_marr3,&
    int_array_3, lbounds=lbound(int_array_3)))

  ! call C code
  call launch_fill_int_array_1(int_marr1)
  call launch_fill_int_array_2(int_marr2)
  call launch_fill_int_array_3(int_marr3)

  ! copy data to host
  call hipCheck(gpufort_mapped_array_copy_data_to_host(int_marr1,c_null_ptr))
  call hipCheck(gpufort_mapped_array_copy_data_to_host(int_marr2,c_null_ptr))
  call hipCheck(gpufort_mapped_array_copy_data_to_host(int_marr3,c_null_ptr))

  ! check
  call hipCheck(hipDeviceSynchronize())
  
  n=0 
  do i = -1,8
    call assert(int_array_1(i) == n)
    n = n +1
  end do 
  n=0 
  do j = -2,7
  do i = -1,8
    call assert(int_array_2(i,j) == n)
    n = n +1
  end do 
  end do 
  n=0 
  do k = -3,6
  do j = -2,7
  do i = -1,8
    call assert(int_array_3(i,j,k) == n)
    n = n +1
  end do 
  end do 
  end do 

  ! destroy
  call hipCheck(gpufort_mapped_array_destroy(int_marr1,c_null_ptr))
  call hipCheck(gpufort_mapped_array_destroy(int_marr2,c_null_ptr))
  call hipCheck(gpufort_mapped_array_destroy(int_marr3,c_null_ptr))
contains

  subroutine assert(condition)
    logical,intent(in) :: condition
    !
    if ( .not. condition ) ERROR STOP "assertion failed"
  end subroutine
end program 
