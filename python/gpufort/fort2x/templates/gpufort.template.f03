module gpufort
  public

contains
  function gpufort_loop_len(first,last,step)
    use iso_c_binding
    implicit none
    integer(c_int),intent(IN) :: first,last,step
    integer(c_int) :: gpufort_loop_len
    !
    integer(c_int) :: len_minus_1
    !
    len_minus_1 = (last-first)/step ! rounds down
    gpufort_loop_len = merge(1+len_minus_1,0,len_minus_1>=0)
  end function
end module
