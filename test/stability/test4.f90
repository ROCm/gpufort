program test1 
  ! logical, parameter :: dosfcflx = .true.   ! not working
  ! logical :: dosfcflx = .true.              ! not working
  logical,parameter :: dosfcflx               ! works!
end program test1
