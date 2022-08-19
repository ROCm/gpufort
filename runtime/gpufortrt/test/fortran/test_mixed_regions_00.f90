program main
  logical(c_bool) :: x(10)
  !$acc init
  !$acc enter data copyin(x)

  !$acc data present(x)
  
  !$acc end data

  !$acc exit data copyout(x)
  !$acc shutdown
end program