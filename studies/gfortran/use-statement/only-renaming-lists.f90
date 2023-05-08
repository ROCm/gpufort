module mymod
  integer, parameter, private :: a = 1
  integer, parameter  :: b = 2
  integer, parameter  :: c = 3
  integer, parameter  :: d = 4
  integer, parameter  :: e = 5
end module

program main

  call only_none()
  call only_renaming()
  call only_only()

contains

subroutine only_none
  !use mymod, only: a ! not found
  use mymod, only: b
  use mymod
  implicit none

  print *, "only list - no list"
  ! print *, a ! not included in any case as private
  print *, b    ! 2
  print *, c    ! 3
  print *, d    ! 4
  print *, e    ! 5
end

subroutine only_renaming
  !use mymod, only: a ! not found
  use mymod, only: b
  use mymod, c => c
  implicit none

  print *, "only list - renaming list"
  ! print *, a ! not included in any case as private
  print *, b    ! 2
  print *, c    ! 3
  print *, d    ! 4
  print *, e    ! 5
end

subroutine only_only
  !use mymod, only: a ! not found
  use mymod, only: b
  use mymod, only: c
  implicit none

  print *, "only list - only list"
  ! print *, a ! not included in any case as private
  print *, b    ! 2
  print *, c    ! 3
  !print *, d   ! not included 
  !print *, e   ! not included
end 

end program main
