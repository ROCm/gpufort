program pointer_to_assumed_size_arrays
  implicit none
  real :: scalar,&
          arr1(2:3),&
          arr2(-1:4,5)

  call abstraction_layer(scalar,arr1,arr2)
contains

subroutine abstraction_layer(scalar,arr1,arr2)
  use iso_c_binding
  implicit none
  real :: scalar,&
          arr1(2:*),&
          arr2(-1:4,*)

  real,pointer :: pscalar,&
                  parr1(:),&
                  parr2(:,:)
  
  pscalar => make_scalar_pointer(scalar)
  print *, c_loc(pscalar)
  parr1  => make_rank1_pointer(arr1,&
                               sizes=[1],&
                               lbounds=lbound(arr1))
  print *, c_loc(parr1)
  print *, shape(parr1)  ! [1] 
  print *, lbound(parr1) ! [2]
  parr2  => make_rank2_pointer(arr2,&
                               sizes=[shape(arr2(:,lbound(arr2,2))),1],&
                               lbounds=lbound(arr2))
  print *, c_loc(parr2)
  print *, shape(parr2)  ! [6,1]
  print *, lbound(parr2) ! [-1,1]
end subroutine

function make_scalar_pointer(arg) result(retval)
  use iso_c_binding
  implicit none
  real,target,intent(in) :: arg
  real,pointer :: retval
  
  print *, c_loc(arg)
  retval => arg
end function

! below can be used to map rank-1 arrays and assumed-size
! arrays to a rank-1 pointer (with size 1)
function make_rank1_pointer(arg,sizes,lbounds) result(retval)
  use iso_c_binding
  implicit none
  real,target,intent(in)      :: arg(*)
  real,pointer                :: retval(:)
  integer,optional,intent(in) :: sizes(1), lbounds(1)
  !
  integer :: opt_sizes(1), opt_lbounds(1)
  !
  opt_sizes   = 1
  opt_lbounds = 1
  if (present(lbounds)) opt_lbounds = lbounds
  
  print *, c_loc(arg)
  call c_f_pointer(c_loc(arg),retval,shape=opt_sizes)
  retval(opt_lbounds(1):) => retval
end function

! below can be used to map rank-2 arrays and assumed-size
! arrays to a rank-2 pointer.
function make_rank2_pointer(arg,sizes,lbounds) result(retval)
  use iso_c_binding
  implicit none
  real,target,intent(in)      :: arg(*)
  integer,optional,intent(in) :: sizes(2), lbounds(2)
  real,pointer                :: retval(:,:)
  !
  integer :: opt_sizes(2), opt_lbounds(2)
  !
  opt_sizes = 1
  if (present(sizes)) opt_sizes = sizes
  opt_lbounds = 1
  if (present(lbounds)) opt_lbounds = lbounds
  
  print *, c_loc(arg)
  call c_f_pointer(c_loc(arg),retval,shape=opt_sizes)
  retval(opt_lbounds(1):,opt_lbounds(2):) => retval
end function

end program 
