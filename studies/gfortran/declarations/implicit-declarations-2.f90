program hello
! default implicit rule

call foo(i) ! i implicitly defined as integer
            ! does not need to be LHS of assignment
            ! just needs to appear

print*, j

contains

subroutine foo(arg)
  integer,intent(in) :: arg ! can be inout,in,out - does not matter
  ! arg = 3 
  print *, arg
end subroutine

end program hello
