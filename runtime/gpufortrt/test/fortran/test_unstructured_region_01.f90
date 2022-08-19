program main
  logical(c_bool) :: x(10)
  !$acc init
  !$acc enter data copyin(x)
  call foo(x)
  !$acc exit data delete(x)
  !$acc shutdown
contains
  subroutine foo(x)
    logical(c_bool) :: x(:)
    !$acc enter data create(x)
    !$acc exit data copyout(x)
  end subroutine
end program