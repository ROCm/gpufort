module mymod1
    interface myprint
       procedure :: myprint_i
       ! or 
       ! module procedure :: myprint_i
    end interface
    
contains
    subroutine myprint_i(arg)
        integer :: arg
        print *, "argument is int"
    end subroutine
end module

module mymod
    use mymod1
contains
    subroutine mysub()
        interface myprint
           procedure :: myprint_r
        end interface
    
        ! now we can use interface myprint for both argument types
        call myprint(i)
        call myprint(r)
    contains
      subroutine myprint_r(arg)
          real :: arg
          print *, "argument is real"
      end subroutine
    end subroutine
end module

program main
    use mymod

    call mysub()
end program main
