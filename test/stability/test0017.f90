program main
  implicit none
  integer i,n

  do
    read(unit,*,END=9)
    i=i+1
  end do
9 n = i/4
end program main
