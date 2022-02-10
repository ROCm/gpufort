! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
program test1 
  ! logical, parameter :: dosfcflx = .true.   ! not working
  ! logical :: dosfcflx = .true.              ! not working
  logical,parameter :: dosfcflx               ! works!
end program test1