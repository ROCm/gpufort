module mytest
  integer, parameter :: a = 1
  integer, parameter :: b = 2
end module

Program Hello
use mytest
use mytest
use mytest, only: a
use mytest, only: b
use mytest, only: a,b

print *, a
print *, b
End Program Hello
