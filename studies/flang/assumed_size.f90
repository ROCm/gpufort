! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
program main
  ! character(1),  parameter :: gnss(*) = ["G","R","E","C"] ! works
  ! character(1),  dimension(*), parameter :: gnss = ["G","R","E","C"] ! works
  ! character(1),  dimension(401:*), parameter :: gnss = ["G","R","E","C"] ! fails
  ! character(1),  parameter :: gnss(401:*) = ["G","R","E","C"] ! fails ! fails
  ! integer, dimension (*), parameter :: foo = (/2, 3, 5/) ! works
  !integer, dimension (1:*), parameter :: foo = (/2, 3, 5/) ! fails
end program