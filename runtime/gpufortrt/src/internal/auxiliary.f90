module gpufortrt_auxiliary
  !> evaluate optional values
  interface eval_optval
     module procedure :: eval_optval_c_bool, eval_optval_c_int, eval_optval_c_size_t
  end interface

contains
  function eval_optval_c_bool(optval,fallback) result(retval)
    implicit none
    use iso_c_binding
    logical(c_bool),intent(in),optional :: optval
    logical(c_bool),intent(in)          :: fallback
    logical(c_bool)                     :: retval
    if ( present(optval) ) then
       retval = optval
    else
       retval = fallback
    endif
  end function

  function eval_optval_c_int(optval,fallback) result(retval)
    implicit none
    use iso_c_binding
    integer(c_int),intent(in),optional :: optval
    integer(c_int),intent(in)          :: fallback
    integer(c_int)                     :: retval
    if ( present(optval) ) then
       retval = optval
    else
       retval = fallback
    endif
  end function
  
  function eval_optval_c_size_t(optval,fallback) result(retval)
    implicit none
    use iso_c_binding
    integer(c_size_t),intent(in),optional :: optval
    integer(c_size_t),intent(in)          :: fallback
    integer(c_size_t)                     :: retval
    if ( present(optval) ) then
       retval = optval
    else
       retval = fallback
    endif
  end function
end module gpufortrt_auxiliary
