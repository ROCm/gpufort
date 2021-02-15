#!/usr/bin/env python3
import sys
from kerneltranslator import * 
from pyparsing import ParseResults

LIST_OF_TENSORS.clear()

testString = "a + d + c + d * (a+c)"
result = arithmeticExpression.parseString(testString)
print(result[0].cStr())

testString = "a(i,sin(y)%x2) = b + d + c + d * (a+c)"
result = assignment.parseString(testString)
print(result[0].cStr())

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
print(result[0].cStr())


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
print(result[0].cStr())

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
print(result[0].cStr())
