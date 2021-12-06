program test1
  integer, parameter :: ims = 12,ime=16,jms=1,jme=10
  REAL, DIMENSION(ims:ime, jms:jme), INTENT(OUT) :: & ! Does not like a comment here
    HEAT2D ! What about here 
end program test1
