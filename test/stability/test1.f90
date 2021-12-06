subroutine external_acc_routine(view)

  implicit none
  real(4), intent(inout) :: view(:,:)

  !$acc data present(view)
  !$acc kernels
  view(1,1) = 4.
  !$acc end kernels
  !$acc end data

end subroutine external_acc_routine
