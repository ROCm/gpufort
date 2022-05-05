! Hiding of a parameter from the parent namespace
! Take a look into test*.f90-gpufort.cpp
module mymod
  integer, parameter :: a = 1

contains

  subroutine mysub()
    integer, parameter :: a = 2
    integer, parameter :: b = 2
    
    contains

    subroutine mysubsub()
      integer, parameter :: a = 3
    integer, parameter   :: b = 3
    end subroutine
  end subroutine
end module

program main
  integer, parameter :: a = 1

contains

  subroutine mysub()
    integer, parameter :: a = 2
  end subroutine
end program
