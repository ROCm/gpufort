! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
program main
  integer :: x = 5
  integer :: b = 3

  associate (x => myfunc(b))
    print *, x
  end associate

contains
  function myfunc(i)
    integer :: i, myfunc
    myfunc = i*2
  end function
end program
