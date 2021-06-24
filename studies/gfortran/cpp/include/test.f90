program main 
! '#' must be at begin of line, 
! no extra tokens are allowed after include statement, this includes comments.
! whitespaces between # and directive name 'include' are allowed
! include depth: 1
#include "snippet1.f90"
! recursive inclusions are allowed
! include depth: 2
#      include "snippet2.f90" 
end program main
