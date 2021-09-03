program main

type, bind(c) :: B
  integer(4) :: mb
end type

type, bind(c) :: A
  type(B) :: tb
end type

interface 

   subroutine read_nested_struct(ta) bind(c,name="read_nested_struct")
     import A 
     type(A) :: ta
   end subroutine

end interface

type(A) :: ta
ta%tb%mb = 251

call read_nested_struct(ta)

end program
