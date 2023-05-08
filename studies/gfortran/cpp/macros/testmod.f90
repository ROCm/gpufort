! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#define used_print_statement1 print *, "I was used."
module testmod
#define used_print_statement2 print *, "I was used too."
end module testmod