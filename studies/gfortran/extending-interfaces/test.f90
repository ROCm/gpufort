! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
module mod1
interface print_value
    module procedure print_value_int
end interface

contains
    subroutine print_value_int(value)
        integer :: value
        print *, value, "is int"
    end subroutine
end module

module mod2
use mod1

interface print_value
    module procedure print_value_real
end interface

contains
    subroutine print_value_real(value)
        real :: value
        print *, value, "is real"
    end subroutine
end module

program main
use mod2
integer :: a  = 1
real :: b = 1.5

call print_value(a)
call print_value(b)

end program main
