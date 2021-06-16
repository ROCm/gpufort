program test1
 use simple
 use nested_subprograms, only: func2
 use complex_types
 implicit none
 
 real                   :: floatScalar
 real(8)                :: doubleScalar
 integer,dimension(:,:) :: intArray2d

 type(mytype) :: t
 
 type(complex_type) :: tc
end test1
