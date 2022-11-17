module iso_c_binding
  integer,parameter :: c_int = 4
  integer,parameter :: c_short = 2
  integer,parameter :: c_long = 8
  integer,parameter :: c_long_long = 8
  integer,parameter :: c_signed_char = 1
  integer,parameter :: c_size_t = 8
  integer,parameter :: c_int8_t = 1
  integer,parameter :: c_int16_t = 2
  integer,parameter :: c_int32_t = 4
  integer,parameter :: c_int64_t = 8
  integer,parameter :: c_int_least8_t = 1
  integer,parameter :: c_int_least16_t = 2
  integer,parameter :: c_int_least32_t = 4
  integer,parameter :: c_int_least64_t = 8
  integer,parameter :: c_int_fast8_t = 1
  integer,parameter :: c_int_fast16_t = 8
  integer,parameter :: c_int_fast32_t = 8
  integer,parameter :: c_int_fast64_t = 8
  integer,parameter :: c_intmax_t = 8
  integer,parameter :: c_intptr_t = 8
  integer,parameter :: c_float = 4
  integer,parameter :: c_double = 8
  integer,parameter :: c_long_double = 10
  integer,parameter :: c_float_complex = 4
  integer,parameter :: c_double_complex = 8
  integer,parameter :: c_long_double_complex = 10
  integer,parameter :: c_bool = 1
  integer,parameter :: c_char = 1
  
  ! GNU extensions
  integer,parameter :: c_int128_t = 16
  integer,parameter :: c_int_least128_t = 16
  integer,parameter :: c_int_fast128_t = 16
  integer,parameter :: c_ptrdiff_t = 8
  integer,parameter :: c_float128 = 16
  integer,parameter :: c_float128_complex = 16
end module
