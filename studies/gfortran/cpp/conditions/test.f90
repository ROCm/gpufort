program main 

! numbers        
#if 0
#error should not be entered
#endif

#if 1
print *, "if 1"
#else
#error should not be entered
#endif

#if 24
print *, "if 24"
#else
#error should not be entered
#endif

#if 2*2==8/2
print *, "if 2*2==8/2"
#elif 2*2==8/2 || 1
#error should not be entered
#else
#error should not be entered
#endif

#define x 4
print *, "#define x 4"

#define x() 5

#if x==5
#error should not be entered
#elif x==4*4/4*1+2-3+1
print *, "elif x==4*4/4*1+2-3+1"
#endif

#undef x
#define x(a,b) (a)*(b)
print *,"#undef x"
print *,"#define x(a,b) (a)*(b)"

#if x(1,5)==5
print *,"x(1,5)==5"
#endif

#define y 3

#if x(1,y)==3
print *,"x(1,y)==y"
#endif

#if defined(x)
print *,"defined(x)"
#endif


! Code below will cause preproc. error
! wrapping any part of if condition in \" ... \" is not valid
! #if "5"
! passing marcro arg wrapped  in \" ... \" is not valid
! #if x("1",5)
! wrapping macro in \" ... \" is not valid
! #if "x(1,5)" == "5"

! Code below causes GCC 7/10/11 seg fault (dev branch not checked yet)
!#if '5'
!print *,"'5'"
!#endif

end program main
