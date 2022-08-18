program main
  use iso_c_binding
  implicit none
  integer,parameter :: m=1,n=3
  logical(c_bool) :: x2(m,n)
  !$acc declare create(x2)

  call foo(x2(:,1),m) ! offset is 1 B
  call foo(x2(:,2),m) ! offset is 1 B
  call foo(x2(:,3),m) ! offset is 1 B
  !call foo(x2(:,4),m) ! offset is 1 B
  !call foo(x2(:,5),m) ! offset is 1 B
contains
  subroutine foo(x1,m)
    use iso_c_binding
    implicit none
    integer,intent(in) :: m
    logical(c_bool),intent(in) :: x1(m)
    !$acc declare copyin(x1)

    !$acc update self(x1)
  end subroutine
end program
