subroutine top_level_subroutine()
  implicit none
  print *, "hallo"
end subroutine

program test1
 use simple
 use nested_subprograms, only: func2
 use complex_types
 implicit none
 
 real                   :: float_scalar
 real(8)                :: double_scalar
 integer,dimension(:,:) :: int_array2d

 type(mytype) :: t
 
 type(complex_type) :: tc

 call top_level_subroutine()
end test1
