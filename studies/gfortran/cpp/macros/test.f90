! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
program main 
use testmod

#define whitespace_in_args(a  , b, c, d ) (a)*(b)*(c)*(d)
#define whitespace_in_subst(a1  , B, c, d )                   (a1)*(B)*(c)*(d)

print *, whitespace_in_args(1,2,3,4)
print *, whitespace_in_subst(1,2,3,4)

#define simple_macro print*,"hallo"
! parameterized macros replace all expressions 
! \b<identifier>\b (\b: word boundary) by the  
! argument string. If there is no word boundary, e.g.
! the expression is 'aa' but the argument is called 'a',
! then the expression will not be replaced by the argument
! string as demonstrated below.
#define parameterized_macro(a) print*,"hallo a aa"
simple_macro
parameterized_macro(user)
! multi-line macros are also supported by the cpp; line break char is '\'
#define multiline_macro(a) print*,"a multiline macro",\
        " line1",\
        " line2"
multiline_macro(test)

! included macros can be used too
#include "snippet.f90" 
included_print_statement

! macros from used modules are not available.
! The following macros from module 'testmod' 
! are not known.
! (commented out because it won't compile)
!used_print_statement1 
!used_print_statement2

end program main