! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
! This file solely exists to extract information about intrinsics via GPUFORT's indexer tool.
! It will not compile with any compiler as non-standard Fortran expressions are used.
module gpufort_intrinsics
contains
#include "gpufort_intrinsics_std_f77.f90"
#include "gpufort_intrinsics_std_f95.f90"
#include "gpufort_intrinsics_std_f2003.f90"
#include "gpufort_intrinsics_std_f2008.f90"
#include "gpufort_intrinsics_std_f2018.f90"
#include "gpufort_intrinsics_std_gnu.f90"
end module
