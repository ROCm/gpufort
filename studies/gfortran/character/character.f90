program main
   use iso_c_binding
   character :: c1
   character(len=2) :: c2
   character(len=3) :: c3
   character(len=4) :: c4
   !character(len=*) :: c_free must be dummy argument
   
   c1 = '123' ! legal, memorizes only first element
   c2 = '123' ! legal, memorizes only first two elements
   c3 = '123' 
   c4 = '123'
   print *, c1 ! only prints 1
   print *, c2 ! only prints 12
   print *, c3 ! prints 123
   print *, c4 ! prints 123
end program main
