program test_optvals
  implicit none
  
  print *, caller()
  print *, caller(optval=.false.)
  print *, caller(optval=.true.)

contains
  
 function caller(optval) result(retval)
   implicit none
   logical,optional,intent(in) :: optval 
   logical                     :: retval
   retval = eval_opt_logical_(optval,.false.)
 end function
 
 function eval_opt_logical_(optval,fallback) result(retval)
   implicit none
   logical,optional,intent(in) :: optval
   logical,intent(in)          :: fallback
   logical                     :: retval
   if ( present(optval) ) then
      retval = optval       
   else
      retval = fallback
   endif
 end function

end program test_optvals
