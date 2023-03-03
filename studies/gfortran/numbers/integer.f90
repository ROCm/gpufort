! This study investigates Fortran's implicit casting behavior.
! Implicit casting is not applied to function arguments.
! Fortran is more strict in this regard than C/C++.
program main
   use iso_c_binding
   integer(c_int) :: i32
   integer(c_long) :: i64

   ! assignments
   i32 = 1 ! i32->i32
   i32 = 1_c_int ! i32->i32
   i32 = 1_c_long ! i64->i32 cast
   i32 = 1e2   ! r32->i32 cast, <num>e<num> is specific notation for float
   i32 = 1.0e2 ! r32->i32 cast, 
   i32 = 1d2   ! r64->i32 cast, <num>d<num> is specific notation for double
   i32 = 1.0d2 ! r64->i32 cast
   i32 = 1d2   ! r64->i32 cast
   i32 = 1.0d2 ! r64->i32 cast

   i64 = 1 ! i32->i64 cast
   i64 = 1_c_int ! i32->i64
   i64 = 1_c_long ! i64->i64
   i64 = 1e2   ! r32->i64 cast 
   i64 = 1.0e2 ! r32->i64 cast
   i64 = 1d2   ! r64->i64 cast
   i64 = 1.0d2 ! r64->i64 cast

   ! arithmetic expressions
   print *, sizeof(1_c_int + 1_c_long) ! implicit upcast r32->r64

   ! procedure arguments
   call fun32(1)
   call fun32(1_c_int)
   !call fun32(1_c_long) ! illegal, no implicit cast i64->i32
   !call fun32(1e2) ! illegal, no implicit cast r32->i32
   !call fun32(1d2) ! illegal, no implicit cast r64->i32
   !call fun64(1) ! illegal, no implicit cast r32->r64
   !call fun64(1_c_int) ! illegal, no implicit cast r32->r64
   call fun64(1_c_long)

contains
  subroutine fun32(arg)
    integer(c_int) :: arg ! 'value' qualifier does not change behavior
  end subroutine

  subroutine fun64(arg)
    integer(c_long) :: arg
  end subroutine
end program
