module params
  parameter (nsize=1000)
end

! example of simple Fortran AMD GPU offloading
program main
  use params, only: nsize
  real :: a(nsize) = 0
  real :: b(nsize) = (/(i, i=1,nsize)/)
  real :: c(nsize) = 10
  integer i

  call foo_target_teams_distribute_parallel_do(a,b,c)
  call foo_target_teams_distribute_parallel_do_composite(a,b,c)
  call foo_target_teams_distribute_parallel_do_simd(a,b,c)
  call foo_target_nested_collapse(a,b,c)

  write(6,*)"a(1)=", a(1), "    a(2)=", a(2)
  if (a(1).ne.11 .or. a(2).ne.22) then
    write(6,*)"ERROR: wrong answers"
    stop 2
  endif
  write(6,*)"Success: if a diagnostic line starting with DEVID was output"
  
  call foo_target_async_wait(a)
  
  return
end

subroutine foo_target_teams_distribute_parallel_do(a,b,c)
  use params, only: nsize
  real a(nsize), b(nsize), c(nsize)
  integer i, n
  n = 0
  !$omp target map(from:a(1:nsize)) map(to:b,c)
  !$omp teams distribute parallel do reduction(+:n)
  do i=1,nsize
    a(i) = b(i) * c(i) + i
    n = n +1
  end do
 !$omp end target
end

subroutine foo_target_teams_distribute_parallel_do_composite(a,b,c)
  use params, only: nsize
  real a(nsize), b(nsize), c(nsize)
  integer i
  !$omp target teams distribute parallel do map(from:a) map(to:b,c) 
  do i=1,nsize
    a(i) = b(i) * c(i) + i
  end do
  ! Optional: 
  !$omp end target teams distribute parallel do
end

subroutine foo_target_teams_distribute_parallel_do_simd(a,b,c)
  use params, only: nsize
  real a(nsize), b(nsize), c(nsize)
  integer i
  !$omp target map(from:a) map(to:b,c) 
  !$omp teams distribute parallel do simd
  do i=1,nsize
    do j=1,nsize
      a(i) = b(i) * c(j) + i
    end do
  end do
  !$omp end target
end

subroutine foo_target_nested_collapse(a,b,c)
  use params, only: nsize
  real a(nsize), b(nsize), c(nsize)
  integer i
  !$omp target map(from:a) map(to:b,c) 
  !$omp teams distribute parallel do collapse(2) 
  do i=1,nsize
    do j=1,nsize
      a(i) = b(i) * c(j) + i
    end do
  end do
  !$omp end target
end

subroutine foo_target_subarrays(a)
  use params, only: nsize
  real a(nsize)
  real, pointer :: a1(:),a2(:)
  integer i

  a1 = a(:nsize/2)
  a2 = a(1+nsize/2:)
  !$omp target teams distribute parallel do map(tofrom:a1,a2) 
  do i=1,nsize
    if ( i .le. nsize/2 ) then
       a1(i) = a1(i)
    else
       a2(i) = a2(i)
    end if 
  end do
  ! Optional: 
  !$omp end target teams distribute parallel do
end

subroutine foo_target_async(a)
  use params, only: nsize
  real a(nsize)
  integer i

  !$omp target teams distribute parallel do nowait map(tofrom:a) depend(in:a) depend(out:a)
  do i=1,nsize
    a(i) = i
  end do
  !$omp target teams distribute parallel do depend(in:a)
  do i=1,nsize
    a(i) = a(i) * 2;
  end do
end

subroutine foo_target_async_wait(a)
  use params, only: nsize
  real    :: a(nsize)
  integer :: i

  !!$omp target teams distribute parallel do nowait map(tofrom:a) depend(in:a) depend(out:a) ! F90-W-0547-OpenMP feature, DEPEND, not yet implemented in this version of the compiler.
  !$omp target teams distribute parallel do nowait map(to:a)
  do i=1,nsize
    a(i) = i
  end do
  !!$omp target teams distribute parallel do nowait depend(in:a) ! F90-W-0547-OpenMP feature, DEPEND, not yet implemented in this version of the compiler.
  !$omp target teams distribute parallel do nowait map(from:a)
  do i=1,nsize
    a(i) = a(i) * 2 
  end do

  !!$omp taskwait depend(out:a) ! F90-S-0034-Syntax error at or near identifier depend
  !$omp taskwait
  !!$omp target update from(a)

  do i=1,nsize
    write(*,*) a(i)
  end do
end
