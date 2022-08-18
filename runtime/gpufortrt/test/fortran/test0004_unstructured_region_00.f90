program main
  logical :: x(10)
  !$acc init
  !$acc enter data copyin(x)
  !$acc exit data copyout(x)
  !$acc shutdown
end program
