program main
  use iso_c_binding
  use iso_fortran_env, only: output_unit
  implicit none

  integer(c_int),parameter :: & ! must be parameter
      lb1 = -1,& ! lower bound dim 1 
      ub1 =  2,& ! upper bound dim 1 
      lb2 = -2,& ! lower bound dim 2 
      ub2 =  3   ! upper bound dim 2 
  integer :: i,j,c

  ! interoperable type with fixed size array as element
  type, bind(c) :: mytype_t
    ! Fortran fixed size array begin
    integer(c_int),dimension(lb1:ub1,lb2:ub2) :: values
    ! Fortran fixed size array end
    ! dope vector begin
    integer(c_int) :: extents(2) = [ub1-lb1+1,ub2-lb2+1] 
    integer(c_int) :: lbounds(2) = [lb1,lb2]
    ! dope vector end
    ! C++ fixed size array end
  end type
  
  interface 
     subroutine pass_to_cpp(arg) bind(c)
       import mytype_t
       type(mytype_t) :: arg
     end subroutine
  end interface
  ! declare the type
  type(mytype_t) :: mytype
  ! mytype%lbounds = lbound(mytype%values) ! this works too

  ! init the type
  c = 1
  write(output_unit,"(a)",advance="yes") "fortran:"
  do i = lb1,ub1
    do j = lb2,ub2
       mytype%values(i,j) = c  
       write(output_unit,"(a,i0,a,i0,a,i0)",advance="yes") "mytype%values(",i,",",j,") = ", mytype%values(i,j)
       c = c + 1
    end do
  end do
  print *, ""
  ! now call the C routine
  call pass_to_cpp(mytype)
end program
