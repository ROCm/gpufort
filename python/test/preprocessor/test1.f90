program main 

#define b 2 
#define c 5
#define size8(x) 8*(x)*b

#if defined(CUDA)
if ( 1 > 0 ) print *, size8(c)
#else
#endif

end program main
