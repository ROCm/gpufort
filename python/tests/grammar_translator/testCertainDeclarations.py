#!/usr/bin/env python3
import addtoplevelpath
import sys
import test
import translator.translator as translator
#import grammar as translator

#print(translator.declaredVariable.parseString("f_d ( : )")[0]._bounds._bounds[0].__dict__)
#print(translator.declaredVariable.parseString("f_d ( -5: )")[0]._bounds._bounds[0].__dict__)
#print(translator.declaredVariable.parseString("f_d ( :5 )")[0]._bounds._bounds[0].__dict__)
#print(translator.declaredVariable.parseString("f_d ( :5:2 )")[0]._bounds._bounds[0].__dict__)
#print(translator.declaredVariable.parseString("f_d ( 1:5:2 )")[0]._bounds._bounds[0].__dict__)

#print(translator.arithmeticExpression.parseString("ldx * ldy * ldz")[0])
#print(translator.declaredVariable.parseString("f_d ( ldx )")[0].fStr())
#print(translator.declaredVariable.parseString("f_d ( ldx, ldy, ldz)")[0].fStr())
#print(translator.declaredVariable.parseString("f_d ( ldx * ldy * ldz )")[0])

print("hallo")
print(translator.declaredVariable.parseString("f_d ( ldx * ldy * ldz )")[0]._bounds.specifiedBounds())
print("hallo")
print(translator.declaredVariable.parseString("f_d ( -k:k, 5 )")[0]._bounds.indexStr("f_d",True))
print("hallo")
print(translator.declaredVariable.parseString("f_d ( -k:k, 5 )")[0]._bounds.indexStr("f_d"))

print(translator.declaredVariable.parseString("i, k, j, err, idir, ip,  ii, jj, istat")[0])
#print("THIS: "+str(translator.declaration.parseString("INTEGER, INTENT(IN) :: isign, ldx, ldy, nx, ny, nzl")[3]))
#print("THIS: "+str(translator.declaration.parseString("INTEGER, INTENT(IN) :: isign")[3]))
#print("THIS: "+str(translator.declaration.parseString("INTEGER :: isign")[3]))

testdata = []
testdata.append("complex(DP),device :: f_d ( ldx * ldy * ldz )")
testdata.append("INTEGER :: i, k, j, err, idir, ip,  ii, jj, istat")

test.run(
   expression     = translator.declaration,
   testdata       = testdata,
   tag            = None,
   raiseException = True
)
   
#print(translator.declaration.parseString(testdata[0])[0].rhs[0])

print(translator.declaration.parseString("complex :: f_d ( : )")[0].arrayBoundVariableNamesFStr("f_d"))
print(translator.declaration.parseString("complex :: f_d ( : )")[0].arrayVariableIndexMacroStr("f_d"))

print(translator.declaration.parseString("complex :: f_d ( :,: )")[0].arrayBoundVariableNamesFStr("f_d"))
print(translator.declaration.parseString("complex :: f_d ( :,: )")[0].arrayVariableIndexMacroStr("f_d"))

print(translator.declaration.parseString("complex :: f_d ( 5,5 )")[0].arrayBoundVariableNamesFStr("f_d"))
print(translator.declaration.parseString("complex :: f_d ( 5,5 )")[0].arrayVariableIndexMacroStr("f_d"))

print(translator.createCodegenContextFromDeclaration(\
        translator.declaration.parseString("complex :: f_d ( 5,5 )")[0]))

print(translator.createCodegenContextFromDeclaration(\
        translator.declaration.parseString("complex,pinned,device,managed,allocatable,pointer :: f_d ( 2:5,-1:5 )")[0]))

context = translator.createCodegenContextFromDeclaration(\
        translator.declaration.parseString("integer, parameter :: a = x , b = c*x, f_d = 5")[0])

print(context)
translator.changeKind(context[0],"8")
print(context)

print(translator.createCodegenContextFromDeclaration(\
        translator.declaration.parseString("complex,pointer :: f_d ( :,-2: )")[0]))
