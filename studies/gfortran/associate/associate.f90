! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
program main
  integer :: x = 5
  integer :: b = 3
  logical :: condition = .true.

  associate (x => myfunc(x,b,condition))
    print *, x
  end associate

contains
  function myfunc(x,b,condition)
    integer,target :: x,b
    logical :: condition
    integer,pointer :: myfunc
    if ( condition ) then
      myfunc => x
    else 
      myfunc => b
    endif
  end function
end program
