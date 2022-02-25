! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
program test3
  integer, parameter :: ims = 12,ime=16,jms=1,jme=10
  REAL, DIMENSION(ims:ime, jms:jme), INTENT(OUT) :: & ! Does not like a comment here
    HEAT2D ! What about here 
end program test3