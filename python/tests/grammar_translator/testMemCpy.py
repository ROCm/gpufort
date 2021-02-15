#!/usr/bin/env python3
import test
import translator.translator

translator.matrixRanges.parseString("(1:n)")

testdata ="""
a(1:n) = b(1:n)
type0%a(1:n) = type1%b(1:n)
""".strip(" ").strip("\n").split("\n")

test.run(
   expression     = translator.memcpy,
   testdata       = testdata,
   tag            = "memcpy",
   raiseException = True
)
v = testdata[0]
print(translator.memcpy.parseString(v)[0].fStr(False,True,8))

assert translator.memcpy.parseString(v)[0].fStr(True,True,8).replace(" ","")   == "hipMemcpy(a,b,1_8*(8)*(n),hipMemcpyDeviceToDevice)"
assert translator.memcpy.parseString(v)[0].fStr(False,True,8).replace(" ","")  == "hipMemcpy(c_loc(a),b,1_8*(8)*(n),hipMemcpyDeviceToHost)"
assert translator.memcpy.parseString(v)[0].fStr(True,False,8).replace(" ","")  == "hipMemcpy(a,c_loc(b),1_8*(8)*(n),hipMemcpyHostToDevice)"
assert translator.memcpy.parseString(v)[0].fStr(False,False,8).replace(" ","") == "hipMemcpy(c_loc(a),c_loc(b),1_8*(8)*(n),hipMemcpyHostToHost)"

v = testdata[1]
assert translator.memcpy.parseString(v)[0].fStr(True,True,8).replace(" ","")   == "hipMemcpy(type0%a,type1%b,1_8*(8)*(n),hipMemcpyDeviceToDevice)"
assert translator.memcpy.parseString(v)[0].fStr(False,True,8).replace(" ","")  == "hipMemcpy(c_loc(type0%a),type1%b,1_8*(8)*(n),hipMemcpyDeviceToHost)"
assert translator.memcpy.parseString(v)[0].fStr(True,False,8).replace(" ","")  == "hipMemcpy(type0%a,c_loc(type1%b),1_8*(8)*(n),hipMemcpyHostToDevice)"
assert translator.memcpy.parseString(v)[0].fStr(False,False,8).replace(" ","") == "hipMemcpy(c_loc(type0%a),c_loc(type1%b),1_8*(8)*(n),hipMemcpyHostToHost)"
#print(translator.memcpy.parseString(v)[0].fStr(True,True,8).replace(" ","")  ) 
#print(translator.memcpy.parseString(v)[0].fStr(False,True,8).replace(" ","") )
#print(translator.memcpy.parseString(v)[0].fStr(True,False,8).replace(" ","") ) 
#print(translator.memcpy.parseString(v)[0].fStr(False,False,8).replace(" ",""))

print("SUCCESS")
