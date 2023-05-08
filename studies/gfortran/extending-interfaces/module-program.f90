module mymod
    interface myprint
       procedure :: myprint_i
       ! or 
       ! procedure :: myprint_i
    end interface
    
contains
    subroutine myprint_i(arg)
        integer :: arg
        print *, "argument is int"
    end subroutine
end module

program main
    use mymod
    ! overloade interface myprint
    interface myprint
        ! we use procedure instead of module procedure
        procedure :: myprint_r
    end interface
    integer :: i = 1
    real :: r = 5.0
    
    ! now we can use interface myprint for both argument types
    call myprint(i)
    call myprint(r)
contains
    subroutine myprint_r(arg)
        real :: arg
        print *, "argument is real"
    end subroutine
end program main
