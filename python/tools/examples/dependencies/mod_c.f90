module mod_c
  implicit none
  use mod_b, only: alpha, beta

  integer, parameter :: gamma = alpha + beta + 1.5

end module 
