module mymod
contains
    SUBROUTINE mysub()
    contains
        subroutine mynested1
        ! end subroutine 
        end subroutine mynested1
        subroutine mynested2
        end subroutine mynested2
        subroutine mynested3()
        ! subroutine mynested4
        end subroutine mynested3
    end subroutine mysub
end module mymod

