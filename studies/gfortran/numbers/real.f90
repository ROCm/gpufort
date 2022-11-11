! This study investigates Fortran's implicit casting behavior.
! Implicit casting is not applied to function arguments.
! Fortran is more strict in this regard than C/C++.
program main
   use iso_c_binding
   real(c_float) :: r32
   real(c_double) :: r64
   
   ! assignments
   r32 = 1
   r32 = 1e2
   r32 = 1.0e2
   r32 = 1d2   ! downcast, <num>d<num> is specific notation for double
   r32 = 1.0d2 ! downcast
   r32 = 1d2   ! downcast
   r32 = 1.0d2 ! downcast
   
   r64 = 1 ! upcast
   r64 = 1e2 ! upcast
   r64 = 1.0e2 ! upcast
   r64 = 1d2 
   r64 = 1.0d2
   
   ! arithmetic expressions
   print *, 1.0e2 + 1.0d2 ! implicit upcast
   
   ! procedure arguments
   call fun32(1.0)
   call fun32(1.0e2)
   ! call fun32(1.0d2) ! illegal, no implicit downcast
   ! call fun64(1.0) ! illegal, no implicit upcast
   ! call fun64(1.0e2) ! illegal, no implicit upcast
   call fun64(1.0d2)
   
contains 
  subroutine fun32(arg)
    real(c_float) :: arg
  end subroutine
  
  subroutine fun64(arg)
    real(c_double) :: arg
  end subroutine
end program
