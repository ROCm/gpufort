program main
  implicit none
  integer i,n
  character(len=*) :: a

  interface
  subroutine foo(arg) bind(c,name="foo")
    integer :: arg
  end subroutine
  end interface

  do
    read(unit,*,END=9)
    i=i+1
    if ( index(a,'b') /= 0 ) i = i*2
  end do
9 n = i/4
end program main
