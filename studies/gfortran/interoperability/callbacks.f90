program callbacks
  use iso_c_binding
  implicit none
  !
  type custom_type_t
    integer, allocatable :: buffer(:)
  end type
  !
  type record_t
    type(c_ptr)    :: ptr        = c_null_ptr
    type(c_funptr) :: destructor = c_null_funptr
  end type
  !
  type(custom_type_t),target :: custom_type
  type(record_t)             :: record
  !
  allocate(custom_type%buffer(10)) 
  record%ptr=c_loc(custom_type)
  record%destructor=c_funloc(destructor)
  call call_destructor(record)
  
  record%destructor = c_null_funptr
  call call_destructor(record)
contains

subroutine destructor(cptr)
  use iso_c_binding
  implicit none
  type(c_ptr),intent(in)      :: cptr
  type(custom_type_t),pointer :: custom_type
  call c_f_pointer(cptr,custom_type)
  print *, "destructor of `custom_type_t`"
  print *, "deallocate buffer of len ",size(custom_type%buffer)
  deallocate(custom_type%buffer)  
end subroutine

subroutine call_destructor(record)
  implicit none
  type(record_t),intent(inout) :: record
  interface
    subroutine destroy(cptr)
      use iso_c_binding
      implicit none
      type(c_ptr),intent(in) :: cptr
    end subroutine
  end interface
  procedure(destroy),pointer :: callback
  if ( c_associated(record%destructor) ) then
    call c_f_procpointer(record%destructor,callback)
    call callback(record%ptr)
  else
    print *, "destructor pointer not c_associated"
  endif 
end subroutine

end program
