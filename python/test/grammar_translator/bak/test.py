#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
import sys
from kerneltranslator import * 

LIST_OF_TENSORS.clear()

#Print (assignment.parseString("y(i,j) = sin(x,y)")[0])
#LIST_OF_TENSORS.append("sin")
#Print (assignment.parseString("y(i,j) = sin(x,y)")[0])
#LIST_OF_TENSORS.clear()
#Print (assignment.parseString("y(i,j) = sin(x,y)")[0])

#sys.exit()

#cufKernelDoVariants="""!$cuf kernel do
#!$cuf kernel do
#!$cuf kernel do (1)
#!$cuf kernel do (1)
#!$cuf kernel do (1) <<<*, *>>>
#!$cuf kernel do (1) <<<*, *>>>
#!$cuf kernel do (1) <<<*,*>>>
#!$cuf kernel do(1)
#!$cuf kernel do(1) <<<,>>>
#!$cuf kernel do(1) <<<*, *>>>
#!$cuf kernel do(1) <<<*,*>>>
#!$cuf kernel do (1) <<<*,*,0,stream>>>
#!$cuf kernel do (1) <<<*,*,0,dfft%a2a_comp>>>
#!$cuf kernel do (1) <<<*,*,0,desc%stream_scatter_yz(1)>>>
#!$cuf kernel do(1)<<<*,*,0,stream>>>
#!$cuf kernel do(1) <<<*,*,0,stream>>>
#!$cuf kernel do(1) <<<*,(16,16,1),0,stream>>>
#!$cuf kernel do (2)
#!$cuf kernel do(2)
#!$cuf kernel do(2)<<<*,*>>>
#!$cuf kernel do(2) <<<,>>>
#!$cuf kernel do(2) <<<*,*>>>
#!$cuf kernel do(2) <<<*,*,0,desc%stream_scatter_yz(iproc3)>>>
#!$cuf kernel do(2) <<<*,*, 0, dfft%bstreams(batch_id)>>>
#!$cuf kernel do (2) <<<*,*,0,stream>>>
#!$cuf kernel do(2) <<<*,*,0,stream>>>
#!$cuf kernel do(3)
#!$cuf kernel do(3) <<<*,*>>>
#!$cuf kernel do(3) <<<*,*, 0, dfft%a2a_comp>>>
#!$cuf kernel do(3) <<<*,*,0,dfft%a2a_comp>>>
#!$cuf kernel do(3) <<<*,(16,16,1), 0, stream>>>
#!$cuf kernel do(4)""".split("\n")

#for v in cufKernelDoVariants:
#  try:
#     print("{} -> {}".format(v,cufKernelDo.parseString(v)))
#  except Exception as e:
#     print("failed to parse {}".format(v))
#     print(e)


print (assignment.parseString("a = 5")[0].cStr())
print (derivedTypeMember.parseString("threadidx%x3")[0])
print (derivedTypeMember.parseString("threadidx%x3")[0].cStr())
print (datatype.parseString("double precision")[0].cStr())
print (declaration.parseString("real(kind=8) :: rhx, rhy")[0].cStr())

testPattern = datatype + Optional(Suppress(",") + Group(delimitedList(attribute)),default=[]) + Literal("::").suppress()
testPattern.setParseAction(WhitespaceList)

print (testPattern.parseString("real(kind=8), device :: ")[0].cStr())
print (declaration.parseString("real(kind=8), device :: rhx, rhy").cStr())
print (declaration.parseString("real, device, parameter :: rhx, rhy").cStr())
print (datatype.parseString("integer(kind=2)").cStr())
print (assignment.parseString("y(i,j) = sin(g(threadidx%x))-180 + g(x,y)")[0].cStr())
print (assignment.parseString("y(i,j) = sin(g(threadidx%x))-180 + g(x,y)")[0].cStr())

testDoWhile="""
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

#print(doWhile.parseString(testDoWhile))


#!$cuf kernel do(2) <<<*,*>>>
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

#parseResult = cufLoopKernel.parseString(testCufLoopKernel) 
#print(testCufLoopKernel)

#import inspect

#print(parseResult[0])
#print(parseResult[0].cStr())
#print(cufLoopKernel.parseString(testCufLoopKernel))

#def gen_dict_extract(key, var):
#    if hasattr(var,'iteritems'):
#        for k, v in var.iteritems():
#            if k == key:
#                yield v
#            if isinstance(v, dict):
#                for result in gen_dict_extract(key, v):
#                    yield result
#            elif isinstance(v, list):
#                for d in v:
#                    for result in gen_dict_extract(key, d):
#                        yield result


testSubroutine="""

subroutine swap(x, y) 
   use hip, ONLY: hipDgemm
   implicit none

   real :: x, y, temp   
   
   temp = x  
   x = y 
   y = temp  
   
end subroutine swap
"""

#print(subroutine.parseString(testSubroutine))

testModule="""module ralala
   use beeer
   implicit none

   real, parameter :: pi = 3.1415926536  
   real, parameter :: e = 2.7182818285 
end module ralala"""

#print(module.parseString(testModule))

testProgram="""
program module_example     
use testmodule      
implicit none     

   real :: x, ePowerx, area, radius 
   x = 2.0
   radius = 7.0
   ePowerx = e ** x
   area = pi * radius**2     
   
   call show_consts() 
   
   #print*, "e raised to the power of 2.0 = ", ePowerx
   #print*, "Area of a circle with radius 7.0 = ", area  
   
end program module_example"""

#print(program.parseString(testProgram))

testFortranFile="""
module testmodule  
implicit none 

   real, parameter :: pi = 3.1415926536  
   real, parameter :: e = 2.7182818285 
   
contains      
   subroutine show_consts()          
      #print*, "Pi = ", pi          
      #print*,  "e = ", e     
   end subroutine show_consts 
   
end module testmodule 

program module_example     
use testmodule      
implicit none     

   real :: x, ePowerx, area, radius 
   x = 2.0
   radius = 7.0
   ePowerx = e ** x
   area = pi * radius**2     
   
   call show_consts() 
   
   #print*, "e raised to the power of 2.0 = ", ePowerx
   #print*, "Area of a circle with radius 7.0 = ", area  
   
end program module_example"""
#for el in program.scanString(testFortranFile):
#  #print(el)
#for el in module.scanString(testFortranFile):
#  #print(el)
