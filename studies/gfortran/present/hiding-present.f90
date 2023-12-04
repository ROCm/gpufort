module mod_calling_present
contains
subroutine calling_present(arg)
  implicit none
  integer,optional :: arg
  if ( present(arg) ) then
    continue
  endif
end subroutine
logical function present(arg)
implicit none
  integer :: arg
  present = .true.
  print *, "function call"
end function
end module

module mod_accessing_local_variable
contains
subroutine accessing_local_variable(arg)
  implicit none
  integer,optional :: arg
  logical :: present(5) = .false.
  if ( .not. present(arg) ) then
     print *, "accessing local variable"
  endif
end subroutine
end module

program main
  use mod_calling_present
  use mod_accessing_local_variable
  implicit none
  
  call calling_present(5) ! prints 'function call'
  call accessing_local_variable(3) ! prints 'accessing_local_variable'
end program
