program legal_expressions
contains
  subroutine test(arg1a,arg1b,&
                  arg2a,&
                  !arg2b,arg2c,arg2d,&
                  arg2e)
    implicit none
    ! inner dimension lower and upper bounds must always
    ! be specified if an assume-sized array ('*') is passed
    ! as argument (see arg2[bcd]).
    ! lower bound of last (='*') dimension may be specified (see arg1b)
    real,target,intent(in) :: arg1a(*)
    real,target,intent(in) :: arg1b(-1:*)
    real,target,intent(in) :: arg2a(1,*)
    !real,target,intent(in) :: arg2b(:,*) ! illegal
    !real,target,intent(in) :: arg2c(:4,*) ! illegal
    !real,target,intent(in) :: arg2d(*,*) ! illegal
    real,target,intent(in) :: arg2e(1:4,*)

    !print *, size(arg2e) ! illegal
    print *, size(arg2e,1)
    !print *, size(arg2e,2) ! illegal
    print *, size(arg2e(:,lbound(arg2e,2)))
    print *, [shape(arg2e(:,lbound(arg2e,2))),1]
    
    !print *, ubound(arg2e) ! illegal
    print *, ubound(arg2e,1)
    !print *, ubound(arg2e,2) ! illegal
    ! print *, ubound(arg2e(:,lbound(arg2e,2))) ! internal compiler error with gcc 9.4.0
  
    print *, lbound(arg2e)
    print *, lbound(arg2e,1)
    print *, lbound(arg2e,2)
  end subroutine
end program
