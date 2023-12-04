! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
module devicelib

contains
   
   attributes(device) subroutine deviceFun(a,x,y,N)
     implicit none
     integer :: N
     real :: x, a
     real,intent(inout) :: y
     y = y + a*x
   end subroutine

end module
