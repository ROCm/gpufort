module test

  use cudafor
  use cublas

  type mytype
    REAL(DP),device,ALLOCATABLE::A_d(:,:)
  end type

  contains

    subroutine test()
        implicit none
        !
        INTEGER :: istat = 0, i
        REAL(DP),ALLOCATABLE::A_h(:,:)
        type(mytype) :: abc 

        allocate(abc%A_d(-4:0,5),A_h(-4:0,5))

        istat = cudaMemcpy(abc%A_d,A_h,25)

        istat = cudaMemcpy(A_h,abc%A_d(-4,-4),25)

        deallocate(abc%A_d,A_h)
       
    end subroutine test

end module test

