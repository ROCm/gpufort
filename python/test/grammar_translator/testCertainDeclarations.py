#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import sys
import test
from gpufort import translator
import addtoplevelpath
from gpufort import grammar

#print(translator.declared_variable.parseString("f_d ( : )")[0]._bounds._bounds[0].__dict__)
#print(translator.declared_variable.parseString("f_d ( -5: )")[0]._bounds._bounds[0].__dict__)
#print(translator.declared_variable.parseString("f_d ( :5 )")[0]._bounds._bounds[0].__dict__)
#print(translator.declared_variable.parseString("f_d ( :5:2 )")[0]._bounds._bounds[0].__dict__)
#print(translator.declared_variable.parseString("f_d ( 1:5:2 )")[0]._bounds._bounds[0].__dict__)

#print(translator.arithmetic_expression.parseString("ldx * ldy * ldz")[0])
#print(translator.declared_variable.parseString("f_d ( ldx )")[0].f_str())
#print(translator.declared_variable.parseString("f_d ( ldx, ldy, ldz)")[0].f_str())
#print(translator.declared_variable.parseString("f_d ( ldx * ldy * ldz )")[0])

print("hallo")
print(translator.declared_variable.parseString("f_d ( ldx * ldy * ldz )")[0]._bounds.specified_bounds())
print("hallo")
print(translator.declared_variable.parseString("f_d ( -k:k, 5 )")[0]._bounds.index_str("f_d",True))
print("hallo")
print(translator.declared_variable.parseString("f_d ( -k:k, 5 )")[0]._bounds.index_str("f_d"))

print(translator.declared_variable.parseString("i, k, j, err, idir, ip,  ii, jj, istat")[0])
#print("THIS: "+str(translator.parse_declaration("INTEGER, INTENT(IN) :: isign, ldx, ldy, nx, ny, nzl")[3]))
#print("THIS: "+str(translator.parse_declaration("INTEGER, INTENT(IN) :: isign")[3]))
#print("THIS: "+str(translator.parse_declaration("INTEGER :: isign")[3]))

testdata = []
testdata.append("complex(DP),device :: f_d ( ldx * ldy * ldz )")
testdata.append("INTEGER :: i, k, j, err, idir, ip,  ii, jj, istat")

test.run(
   expression     = translator.declaration,
   testdata       = testdata,
   tag            = None,
   raiseException = True
)
   
#print(translator.parse_declaration(testdata[0])[0].rhs[0])

print(translator.create_index_records_from_declaration(\
        translator.parse_declaration("complex :: f_d ( 5,5 )")[0]))

print(translator.create_index_records_from_declaration(\
        translator.parse_declaration("complex,pinned,device,managed,allocatable,pointer :: f_d ( 2:5,-1:5 )")[0]))

context = translator.create_index_records_from_declaration(\
        translator.parse_declaration("integer, parameter :: a = x , b = c*x, f_d = 5")[0])

print(context)
translator.change_kind(context[0],"8")
print(context)

print(translator.create_index_records_from_declaration(\
        translator.parse_declaration("complex,pointer :: f_d ( :,-2: )")[0]))