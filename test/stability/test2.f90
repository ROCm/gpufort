module allocate_unified_module

interface

  subroutine allocate_unified_impl( a, N ) bind(C,name="allocate_unified_impl")
    use, intrinsic :: iso_c_binding
    type(c_ptr) :: a
    integer(c_int), value :: N
  end subroutine

end interface

contains

  subroutine allocate_unified( a, N )
    use, intrinsic :: iso_c_binding
    real(c_double), pointer :: a(:)
    integer(c_int) :: N
    type(c_ptr) :: value_cptr
    call allocate_unified_impl(value_cptr, N)
    call c_f_pointer(value_cptr,a,(/N/))
  end subroutine

end module
