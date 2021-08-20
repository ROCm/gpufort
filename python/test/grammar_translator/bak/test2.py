#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
import sys
from kerneltranslator import * 
from pyparsing import ParseResults

LIST_OF_TENSORS.clear()

testString = "a + d + c + d * (a+c)"
result = arithmetic_expression.parseString(testString)
print(result[0].c_str())

testString = "a(i,sin(y)%x2) = b + d + c + d * (a+c)"
result = assignment.parseString(testString)
print(result[0].c_str())

testDo="""
do i=1,n
do j=1,n
  a(i,j) = 5*b(i,j)
  if ( a < 5 .and. x > 3 ) then
    do k=1,n
      b=5
    enddo
  endif
enddo
enddo"""

result = doLoop.parseString(testDo)
print(result[0].c_str())


testWhile="""
do while (k < 3)
do i=1,n
do j=1,n
  a(i,j) = 5*b(i,j)
  if ( a < 5 .and. x > 3 ) then
    do k=1,n
      b=5
    enddo
  endif
enddo
enddo
enddo"""

result = whileLoop.parseString(testWhile)
print(result[0].c_str())

testCufLoopKernel="""
!$cuf kernel do(2) <<<*,*, 0, dfft%bstreams(batch_id)>>>
do i=1,n
do j=1,n
  a(i,j) = 5*b(i,j)
  if ( a < 5 .and. x > 3 ) then
    do k=1,n
      b=5
    enddo
  else if ( a(i,j) .ge. .TRUE.) then
    print *, "else-if"
  else
    print *, "else"
  endif
  ! this comment is ignored
  call mysubroutine(i,j)
enddo
enddo"""

#print(testCufKernels)
result = cufLoopKernel.parseString(testCufLoopKernel)
print(result[0].c_str())