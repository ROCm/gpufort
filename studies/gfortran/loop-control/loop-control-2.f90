program loop_control

! From the GCC online docs:

!The CYCLE and EXIT statements specify that the remaining statements in the current iteration of a particular active (enclosing) DO loop are to be skipped.
!CYCLE specifies that these statements are skipped, but the END DO statement that marks the end of the DO loop be executed—that is, the next iteration, if any, is to be started. If the statement marking the end of the DO loop is not END DO—in other words, if the loop is not a block DO—the CYCLE statement does not execute that statement, but does start the next iteration (if any).
!EXIT specifies that the loop specified by the DO construct is terminated.

!The CONTINUE statement is often used as a place to hang a statement label, usually it is the end of a DO loop.
!The CONTINUE statement is used primarily as a convenient point for placing a statement label, particularly as the terminal statement in a DO loop. Execution of a CONTINUE statement has no effect.
!If the CONTINUE statement is used as the terminal statement of a DO loop, the next statement executed depends on the DO loop exit condition.

print *, "scenario 2 (Shared Loop Label):"
do 50 i=1,3
  print *, "start inner"
  do 50 j=1,10
    print *, "i,j=", i, ",", j
    if ( j == 1 ) then
      print *, "continue as j == 1"
      continue ! expect j = 2 afterwards
      print *, "continue is C++ empty statement, so this will be printed"
    else if ( j == 3 ) then
      print *, "cycle as j == 3"
      cycle ! expect j = 4 afterwards
      print *, "cycle is C++ continue, so this will not be printed"
    else if ( j == 5 ) then
      print *, "exit as j == 5"
      exit
      print *, "exit is C++ break, so this will not be printed"
    endif
50 continue
print *, "end outer"

end program loop_control
