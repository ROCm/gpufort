program main 

#define b 2 
#define c 5
#define size8(x) 8*(x)*b

#if defined(CUDA)
#  if defined(CUDA1)
#  elif defined(CUDA3)
#  elif defined(CUDA2)
if ( 1 > 0 ) print *, size8(c)
#  endif
#else
#endif


#if defined(HIP)
#  if defined(CUDA1)
#  elif defined(CUDA3)
#  elif defined(CUDA2)
if ( 1 > 0 ) print *, size8(2*c+c)
#  endif
#else
print *, "else"
#endif

end program main
