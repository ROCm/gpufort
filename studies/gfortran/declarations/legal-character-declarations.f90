program character_declarations
  use iso_c_binding
  
  !
  ! F77
  !
  
  ! The following are all illegal as only 
  ! character[*<len>|*(<len>)]
  ! is allowed:  
  ! character*(kind=c_char,len=1) :: f77_c(3,4)
  ! character*(len=2,kind=c_char) :: f77_c(3,4) 
  ! character(len=3,c_char) :: c(3,4)*4 
  ! character*(3,kind=c_char) :: f77_c(3,4)
  ! character*(4,c_char) :: f77_c(3,4)*4
  ! character*(4,kind=c_char) :: f77_c(3,4)*5 
  
  character :: f77_c1a(3,4) ! len=1
  ! character*() :: foo(3,4) illegal
  character*1 :: f77_c1b(3,4) ! 
  ! character*(len=1) :: f77_c1(3,4) ! illegal
  ! character*(len=1) :: f77_c1(3,4) ! illegal
  character*(2) :: f77_c2a(3,4) ! 
  ! character*2*1 :: f77_c2b(3,4) !  illegal
  character*(2*1) :: f77_c2b(3,4) ! len expression can be any arithmetic parameter expression
  character*(2) :: f77_c3(3,4)*3 ! *3 overwrites len in brackets
  
  !
  ! Modern Fortran
  !

  ! character parameter list takes an additional kind argument, which
  ! can be a named argument. If both kind and len are named arguments,
  ! the order of arguments can be interchanged.
  
  character :: c1a(3,4)
  !character() :: foo(3,4) illegal
  character(kind=c_char) :: c1b(3,4)
  character(kind=c_char,len=1) :: c1c(3,4)
  character(2) :: c2a(3,4)
  character(len=2) :: c2b(3,4)
  character(len=2,kind=c_char) :: c2c(3,4) ! swapped
  !character(len=3,c_char) :: c3(3,4)*4 ! illegal to have named arg before unnamed
  character(3,kind=c_char) :: c3(3,4)
  character(4,c_char) :: c4(3,4)
  character(4,c_char) :: c5(3,4)*5 ! *5 overwrites len in brackets
  ! Below is only legal for parameters or procedure dummy arguments
  character(*),parameter :: c6a = "123456"
  character(len=*),parameter :: c6b = "123456"
  ! Below is only legal for allocatable/pointer types
  character(:),allocatable,target :: c7a
  character(:),pointer :: c7b
  character(len=:),allocatable :: c7c

  c7a = "1234567"
  c7b => c7a
  allocate(c7c,source=c7a)
 
  print *, "len(f77_c1a)=",len(f77_c1a)
  print *, "len(f77_c1b)=",len(f77_c1b)
  print *, "len(f77_c2a)=",len(f77_c2a)
  print *, "len(f77_c2b)=",len(f77_c2b)
  print *, "len(f77_c3) =",len(f77_c3)
  
  print *, "len(c1a)    =",len(c1a)
  print *, "len(c1b)    =",len(c1b)
  print *, "len(c1c)    =",len(c1c)
  print *, "len(c2a)    =",len(c2a)
  print *, "len(c2b)    =",len(c2b)
  print *, "len(c2c)    =",len(c2c)
  print *, "len(c3)     =",len(c3)
  print *, "len(c4)     =",len(c4)
  print *, "len(c5)     =",len(c5)
  print *, "len(c6a)     =",len(c6a)
  print *, "len(c6b)     =",len(c6b)
  print *, "len(c7a)     =",len(c7a)
  print *, "len(c7b)     =",len(c7b)
  print *, "len(c7c)     =",len(c7c)

contains

subroutine mysub(arg1,arg2,arg3,arg4)
  character(*) :: arg1
  character(len=*) :: arg2
  !character(-2:*) :: arg1 ! illegal
  !character(len=-2:*) :: arg2 ! illegal
end subroutine
end program
