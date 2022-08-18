program main
  use iso_c_binding
  implicit none
  integer,parameter :: m=1,n=3
  logical :: x2(m,n)
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
    logical,intent(in) :: x1(m)
    !$acc declare copyin(x1)

    !$acc kernels
    x1 = 1
    !$acc end kernels
  end subroutine
end program
