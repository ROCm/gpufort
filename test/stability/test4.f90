! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
program test4 
  logical, parameter :: dosfcflx1 = .true.
  logical :: dosfcflx2 = .true.           
  logical,parameter :: dosfcflx3 = dosfcflx1
end program test4
