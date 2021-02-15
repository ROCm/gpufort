module test

  use cudafor
  use cublas

  contains

    attributes(global) subroutine test_gpu(A_d,N)
        implicit none
        !
        INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
        integer, intent(in) :: N
        real(DP), intent(out) :: A_d(N)
        !
        integer :: i
 
        i = 1 + threadIdx%x + blockIdx%x * blockDim%x
        if ( i <= 5 ) then
           A_d(i) = A_d(i) + 1
        endif
    end subroutine test_gpu

    subroutine test()
        use my_pointer_lib, ONLY: included_pointer_d
        implicit none
        !
        INTEGER :: istat = 0, i
        INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
        REAL(DP),device,ALLOCATABLE::A_d(:)
        REAL(DP),ALLOCATABLE :: A_h(:)
        COMPLEX(kind=8),DEVICE,POINTER :: pointer_d(:)
        type(dim3):: grid, tBlock

        allocate(A_d(5),A_h(5))

        DO i = 1, 5 
           A_h(i) = i+1
        END DO

        istat = cudaMemcpy(A_d,A_h,5)

        !$cuf kernel do(1)
        DO i = 1, 5
           A_d(i) = A_d(i) - 1
        END DO
  
        tBlock = dim3(5,1,1)
        grid = dim3(1,1,1)
        CALL test_gpu<<<grid,tBlock>>>(A_d,5)

        istat = cudaMemcpy(A_h,A_d,5)

        if ( allocated(A_d) ) deallocate(A_d)
        if ( .not. allocated(A_d) ) allocate(A_d(5))
        if ( allocated(A_h) ) deallocate(A_h)
        if ( .not. allocated(A_h) ) allocate(A_h(5))
        deallocate(A_d,A_h)

        pointer_d => included_pointer_d 
       
    end subroutine test

end module test

