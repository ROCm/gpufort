! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
subroutine top_level_subroutine()
  implicit none
  print *, "hallo"
end subroutine

program test1
 use simple
 use nested_procedures, only: func2
 use complex_types
 use private_mod1
 implicit none
 
 real                   :: float_scalar
 real(8)                :: double_scalar
 integer,dimension(:,:) :: int_array2d

 type(mytype) :: t
 
 type(complex_type) :: tc

 call top_level_subroutine()
end test1
