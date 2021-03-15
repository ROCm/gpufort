module runtime
  implicit none

  integer,parameter,save               :: N = 10
  real,allocatable,save,dimension(:,:) :: a
  !$omp declare target to(a) 
  
  real,allocatable,save,dimension(:,:) :: b

contains
  subroutine init_device_data()
    implicit none
    allocate(a(N,N))
    allocate(b(N,N))
    !$omp target enter data map(alloc:b)
  end subroutine
  
  subroutine free_device_data()
    implicit none
    deallocate(a)
    !$omp target exit data map(release:b)
    deallocate(b)
  end subroutine

end module

program main
  use runtime
  implicit none
  integer :: i,j
  !
  call init_device_data()

  ! on host
  print *, allocated(a)
  !$omp parallel do collapse(2)
  do j = 1,N
    do i = 1,N
      a(i,j) = (i-1) + (j-1)*N 
      b(i,j) = 0.0002
    end do
  end do
  
  ! on target
  !$omp target parallel do collapse(2)
  do j = 1,N
    do i = 1,N
      a(i,j) = 2*a(i,j) + 0.001 + b(i,j) ! a is synchronized but device 'b' was not updated, so b(i,j) is 0
    end do
  end do
  
  do j = 1,N
    do i = 1,N
      Print *, a(i,j)
    end do
  end do
  !
  call free_device_data()
end program
