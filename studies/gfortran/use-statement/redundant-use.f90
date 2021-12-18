! SPDX-License-Identifier: MIT
! Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
module mytest
  integer, parameter :: a = 1
  integer, parameter :: b = 2
end module

Program Hello
use mytest
use mytest
use mytest, only: a
use mytest, only: b
use mytest, only: a,b

print *, a
print *, b
End Program Hello