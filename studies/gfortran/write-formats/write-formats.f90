module output_mod
  private
  public    :: print_val
  
  !interface print_type
  !  module procedure :: &
  !    print_type_logical,&
  !    print_type_short,&
  !    print_type_int,&
  !    print_type_long,&
  !    print_type_float,&
  !    print_type_double,&
  !    print_type_float_complex,&
  !    print_type_double_complex
  !end interface
  
  !interface print_kind
  !  module procedure :: &
  !    print_kind_logical,&
  !    print_kind_short,&
  !    print_kind_int,&
  !    print_kind_long,&
  !    print_kind_float,&
  !    print_kind_double,&
  !    print_kind_float_complex,&
  !    print_kind_double_complex
  !end interface
  
  !interface print_bytes
  !  module procedure :: &
  !    print_bytes_logical,&
  !    print_bytes_short,&
  !    print_bytes_int,&
  !    print_bytes_long,&
  !    print_bytes_float,&
  !    print_bytes_double,&
  !    print_bytes_float_complex,&
  !    print_bytes_double_complex
  !end interface

  interface print_val
    module procedure :: &
      print_val_bool,&
      print_val_short,&
      print_val_int,&
      print_val_int_arr,&
      print_val_long,&
      print_val_float,&
      print_val_double,&
      print_val_float_complex,&
      print_val_double_complex
  end interface
contains
  subroutine print_val_bool(val)
    use iso_c_binding
    use iso_fortran_env
    implicit none
    logical(c_bool) :: val
    if ( val ) then
      write( output_unit, "(a)", advance="no") "true"
    else
      write( output_unit, "(a)", advance="no") "false"
    endif
  end subroutine
  
  subroutine print_val_short(val)
    use iso_c_binding
    use iso_fortran_env
    implicit none
    integer(c_short) :: val
    write( output_unit, "(i0)", advance="no") val
  end subroutine
  
  subroutine print_val_int(val)
    use iso_c_binding
    use iso_fortran_env
    implicit none
    integer(c_int) :: val
    write( output_unit, "(i0)", advance="no") val
  end subroutine
  
  subroutine print_val_int_arr(val)
    use iso_c_binding
    use iso_fortran_env
    implicit none
    integer(c_int),dimension(:) :: val
    integer :: i1
    write( output_unit, "(a)", advance="no" ) "("
    do i1 = lbound(val,1),ubound(val,1)
      if ( i1 == ubound(val,1) ) then
          write( output_unit, "(i0)", advance="no") val(i1)
      else
          write( output_unit, "(i0,a)", advance="no") val(i1), ","
      endif
    enddo
    write( output_unit, "(a)", advance="no" ) ")"
  end subroutine
  
  subroutine print_val_long(val)
    use iso_c_binding
    use iso_fortran_env
    implicit none
    integer(c_long) :: val
    write( output_unit, "(i0)", advance="no") val
  end subroutine
  
  subroutine print_val_float(val)
    use iso_c_binding
    use iso_fortran_env
    implicit none
    real(c_float) :: val
    write( output_unit, "(e13.6e2)", advance="no") val
  end subroutine
  
  subroutine print_val_double(val)
    use iso_c_binding
    use iso_fortran_env
    implicit none
    real(c_double) :: val
    write( output_unit, "(e23.15e3)", advance="no") val
  end subroutine
  
  subroutine print_val_float_complex(val)
    use iso_c_binding
    use iso_fortran_env
    implicit none
    complex(c_float_complex) :: val
    write( output_unit, "(a)", advance="no" ) "("
    write( output_unit, "(:,e13.6e2,:,a:,e13.6e2,:)", advance="no") real(val,kind=c_float),",",aimag(val)
    write( output_unit, "(a)", advance="no" ) ")"
  end subroutine
  
  subroutine print_val_double_complex(val)
    use iso_c_binding
    use iso_fortran_env
    implicit none
    complex(c_double_complex) :: val
    write( output_unit, "(a)", advance="no" ) "("
    write( output_unit, "(e23.15e3,a,e23.15e3)", advance="no") real(val,kind=c_double),",",dimag(val)
    write( output_unit, "(a)", advance="no" ) ")"
  end subroutine
end module

program main
  use iso_c_binding
  use iso_fortran_env
  use output_mod
  implicit none

  ! Variable declarations
  logical(c_bool) :: bool_true = .true.
  logical(c_bool) :: bool_false = .false.
  integer(c_short) :: short = 1_c_short
  integer(c_int),dimension(5) :: int_arr = [1,2,3,4,5]
  real (c_float)  :: float = 1.000001234567890_c_float
  real (c_double) :: double = 1.000001234567890_c_double
  complex (c_float_complex) :: float_cmplx = (1.000001234567890_c_float,2.000001234567890e-3_c_float)
  complex (c_double_complex) :: double_cmplx = (1.000001234567890_c_double,2.000001234567890e-3_c_double)
  
  write( output_unit, '(2a)') &
  'info:compiler version: ', compiler_version()
  write( output_unit, '(2a)') &
  'info:compiler options: ', compiler_options()
  
  call print_val(bool_true)
  print*,""
  call print_val(bool_false)
  print*,""
  call print_val(short)
  print*,""
  call print_val(int_arr)
  print*,""
  call print_val(float)
  print*,""
  call print_val(double)
  print*,""
  call print_val(float)
  print*,""
  call print_val(double)
  print*,""
  call print_val(float_cmplx)
  print*,""
  call print_val(double_cmplx)
  print*,""

end program main
