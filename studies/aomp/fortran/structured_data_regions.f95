program main
  implicit none
  integer,parameter :: n = 10
  real :: a(n,n)
  integer :: i,j
  
  ! host code
  !$omp parallel workshare
  a(:,:) = 1
  !$omp end parallel workshare

  !$omp target data map(tofrom:a)
  !$omp target data
  !$omp target data

  ! not supported as it is not detected as loop
  !!$omp target teams distribute parallel workshare
  !a(:,:) = 1
  !!$omp end target teams distribute parallel workshare
  
  !$omp target teams distribute parallel do collapse(2)
  do j = 1,n
    do i = 1,n
      a(i,j) = i*1+j*n
    end do
  end do
  !$omp end target teams distribute parallel do

  !$omp end target data
  !$omp end target data
  !$omp end target data

  do j = 1,n
    do i = 1,n
      print*, a(i,j)
    end do
  end do

end program main
