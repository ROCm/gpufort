program main
  use iso_c_binding
  !$acc init
  !$acc data
  !$acc end data
  !$acc shutdown
end program
