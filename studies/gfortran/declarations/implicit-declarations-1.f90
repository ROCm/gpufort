MODULE mymod
  ! e = 3 ! assignment statements not allowed in module (if no parameter)
  Parameter(e=3) ! legal, implicitly defined as real(c_float) 
end module

program main
use iso_c_binding
implicit integer(c_float) (a-c)

a = 5
! b = [1,2,3] ! illegal, must be rank 0

call foo()

contains

subroutine foo()
   use iso_c_binding
   implicit real(c_double) (a-c)
   ! variable 'a' is implicitly defined in 'main' as integer
   a = 3.123 ! rhs '3.0' is casted to integer  (floored)
   c = 9.123 ! no local implicit rule is used, implicitly defined as real(8)
   print *, a ! prints '3'
   print *, c_sizeof(a) ! prints '4'
   print *, c ! prints '9.123...'
   print *, c_sizeof(c) ! prints '8'
end subroutine

end program
