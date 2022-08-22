program main
  use iso_c_binding
  logical(c_bool) :: x(10)
  !$acc init
  !$acc enter data copyin(x) async(1)
  call foo(x)
  !$acc exit data delete(x) async(1)
  !$acc shutdown
contains
  subroutine foo(x)
    logical(c_bool) :: x(:)
    !$acc enter data create(x) async(1)
    !$acc exit data copyout(x) async(1)
  end subroutine
end program