REAL FUNCTION lagrange_quad ( x , x0, x1, x2, y0, y1, y2 )

  IMPLICIT NONE

  REAL :: x , x0, x1, x2, y0, y1, y2, c
  PARAMETER c = 5.0
  !$acc routine seq

  lagrange_quad = &
         (x-x1)*(x-x2)*y0 / ( (x0-x1)*(x0-x2) ) + &
         (x-x0)*(x-x2)*y1 / ( (x1-x0)*(x1-x2) ) + &
         (x-x0)*(x-x1)*y2 / ( (x2-x0)*(x2-x1) )

END FUNCTION lagrange_quad
