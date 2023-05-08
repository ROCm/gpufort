module mymod
  integer, parameter, private :: b = 3
  integer, parameter, public  :: a = b
end module

program main
  use mymod
  print *,a
end program
