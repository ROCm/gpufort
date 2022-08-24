! demonstrates: looking up members of private types 
!               where the latter are part of a public type of a module.
module mod1
  private :: private_type
  type :: private_type
    integer,pointer :: var(:)
  end type
  
  type :: public_type
    type(private_type) :: member
  end type
end module

program main
  use mod1
  type(public_type) :: type1
  !
  allocate(type1%member%var(5))
  type1%member%var(:) = 1  
end program main
