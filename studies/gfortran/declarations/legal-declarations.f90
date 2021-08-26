program main

type test
  integer a
  integer(4) b
  integer*4 c
  integer(kind(4)) d
  integer(kind=4) e
  integer :: f
  integer(4) :: g
  integer*4 :: h
  integer(kind(4)) :: i
  integer(kind=4) :: j
  !integer, pointer k ! illegal
  integer, pointer :: l
end type


type(test) A
type(test) :: B
! type*test C             ! illegal
! type*test :: D          ! illegal
! type(kind=test) J       ! illegal
! type(kind=test) :: K    ! illegal

end program
