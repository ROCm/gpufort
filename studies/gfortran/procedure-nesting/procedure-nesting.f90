! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
! In program units such as 'program', 'subroutine' There must be only one `contains` statement
! and it must be in a top-level module,program, or procedure.
!
! Two levels of nesting (via contains) are only supported within a
! module as a module is not a program unit but instead a collection of program units
! (and interfaces, global variables/parameters, derived type definitions.)

module test
contains 
  subroutine mod_level_1_proc()
     Print *, "mod_level_1_proc"
  contains 
     subroutine mod_level_2_proc()
       Print *, "mod_level_2_proc"
     !contains ! a third level is not supported
     !  subroutine level_3_proc()
     !    Print *, "level_3_proc"
     !  end subroutine
     end subroutine
  end subroutine
end module test

subroutine level_1_proc()
     Print *, "level_1_proc"
     call level_2_proc_1 ! call nested subroutine
  contains 
     subroutine level_2_proc_1()
       Print *, "level_2_proc_1"
       call level_2_proc_2() ! call other nested subroutine defined below; order is not important
     end subroutine
     subroutine level_2_proc_2()
       Print *, "level_2_proc_2"
     end subroutine
end subroutine

Program Hello
  call level_1_proc()
  ! call level_2_proc_1() ! is undefined
  contains 
  subroutine prog_level_1_proc()
     Print *, "prog_level_1_proc"
  !contains 
  !   subroutine prog_level_2_proc_1()
  !     Print *, "prog_level_2_proc_1"
  !   end subroutine
  end subroutine
End Program Hello