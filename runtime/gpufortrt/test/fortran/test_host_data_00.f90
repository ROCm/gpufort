program main
  use iso_c_binding
  implicit none
  logical(c_bool) :: x(10)
  logical(c_bool) :: y(10,20)
  !$acc init
  !$acc data copyin(x,y)
  call foo(x,y,.true.)
  call foo(x,y,.false.)
  !$acc end data
  !$acc shutdown
contains
   subroutine foo(x,y,use_gpu)
     implicit none
     logical(c_bool),intent(in) :: x(*)
     logical(c_bool),intent(in) :: y(1:10,2:*)
     logical,intent(in) :: use_gpu
     !$acc host_data use_device(x,y) if_present if(use_gpu)
     print *, lbound(x)
     print *, lbound(y)
     !$acc end host_data
   end subroutine
end program
