#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
import addtoplevelpath
import sys
import test
import translator.translator as translator
#import grammar as translator

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
#print("THIS: "+str(translator.fortran_declaration.parseString("INTEGER, INTENT(IN) :: isign, ldx, ldy, nx, ny, nzl")[3]))
#print("THIS: "+str(translator.fortran_declaration.parseString("INTEGER, INTENT(IN) :: isign")[3]))
#print("THIS: "+str(translator.fortran_declaration.parseString("INTEGER :: isign")[3]))

testdata = []
testdata.append("complex(DP),device :: f_d ( ldx * ldy * ldz )")
testdata.append("INTEGER :: i, k, j, err, idir, ip,  ii, jj, istat")

test.run(
   expression     = translator.declaration,
   testdata       = testdata,
   tag            = None,
   raiseException = True
)
   
#print(translator.fortran_declaration.parseString(testdata[0])[0].rhs[0])

print(translator.fortran_declaration.parseString("complex :: f_d ( : )")[0].array_bound_variable_names_f_str("f_d"))
print(translator.fortran_declaration.parseString("complex :: f_d ( : )")[0].array_variable_index_macro_str("f_d"))

print(translator.fortran_declaration.parseString("complex :: f_d ( :,: )")[0].array_bound_variable_names_f_str("f_d"))
print(translator.fortran_declaration.parseString("complex :: f_d ( :,: )")[0].array_variable_index_macro_str("f_d"))

print(translator.fortran_declaration.parseString("complex :: f_d ( 5,5 )")[0].array_bound_variable_names_f_str("f_d"))
print(translator.fortran_declaration.parseString("complex :: f_d ( 5,5 )")[0].array_variable_index_macro_str("f_d"))

print(translator.create_index_records_from_declaration(\
        translator.fortran_declaration.parseString("complex :: f_d ( 5,5 )")[0]))

print(translator.create_index_records_from_declaration(\
        translator.fortran_declaration.parseString("complex,pinned,device,managed,allocatable,pointer :: f_d ( 2:5,-1:5 )")[0]))

context = translator.create_index_records_from_declaration(\
        translator.fortran_declaration.parseString("integer, parameter :: a = x , b = c*x, f_d = 5")[0])

print(context)
translator.change_kind(context[0],"8")
print(context)

print(translator.create_index_records_from_declaration(\
        translator.fortran_declaration.parseString("complex,pointer :: f_d ( :,-2: )")[0]))