
subroutine top_level_subroutine()
  ! default rule: implicit integer (i-n), real (a-h,o-z)
  print *, "hallo"

  PARAMETER(i = 3) 
  PARAMETER(n = 3) 

  PARAMETER(a = 1.0, o = 3.0) 
end subroutine

program test1
 use simple
 use nested_procedures, only: func2
 use complex_types
 use private_mod1
 implicit integer (a-c), real (i-n)
 
 real                   :: float_scalar
 real(8)                :: double_scalar
 integer,dimension(:,:) :: int_array2d
 PARAMETER(a = 3) 
 PARAMETER(c = 5) 
 PARAMETER(i = 1.0, n = 2.0) 

 type(mytype) :: t
 
 type(complex_type) :: tc

 call top_level_subroutine()
end test1
