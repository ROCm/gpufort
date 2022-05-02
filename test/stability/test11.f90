program main
  integer i,j
  integer,parameter :: ntasks_local = 4, tasks_per_split = 2
  integer :: icolor( tasks_per_split, tasks_per_split ) 
 
  j = 0
  DO WHILE ( j .LT. ntasks_local / tasks_per_split )
    DO i = 1, tasks_per_split
      icolor( i + j * tasks_per_split ) = j
    END DO
    j = j + 1
  END DO
end program
