module mymod
!  REAL C,F ! not legal in module
!  C(F) = 5.0*(F - 32.0)/9.0
end module mymod

program main
  integer,parameter :: m = -1, n = -5
  REAL C,F
  C(F) = 5.0*(F - 32.0)/9.0 ! legal and is in scope of sub routine

  print *, "main: ", C(5.0)

  call mysub1()
  call mysub2()
contains 

  subroutine mysub1()
    print *,"mysub1: inheriting definition of program: ",C(5.0)
  end subroutine mysub1
  
  subroutine mysub2()
    REAL C,F
    C(F) = 1.0*(F - 32.0)/9.0 ! legal and is in scope of sub routine
    print *, "mysub2: hiding definition of program: ",C(5.0)
  end subroutine mysub2

end program main
