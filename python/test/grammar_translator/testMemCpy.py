#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import test
from gpufort import translator

translator.matrix_ranges.parseString("(1:n)")

testdata ="""
a(1:n) = b(1:n)
type0%a(1:n) = type1%b(1:n)
""".strip(" ").strip("\n").splitlines()

test.run(
   expression     = translator.memcpy,
   testdata       = testdata,
   tag            = "memcpy",
   raiseException = True
)
v = testdata[0]
print(translator.memcpy.parseString(v)[0].f_str(False,True,8))

assert translator.memcpy.parseString(v)[0].f_str(True,True,8).replace(" ","")   == "hipMemcpy(a,b,1_8*(8)*(n),hipMemcpyDeviceToDevice)"
assert translator.memcpy.parseString(v)[0].f_str(False,True,8).replace(" ","")  == "hipMemcpy(c_loc(a),b,1_8*(8)*(n),hipMemcpyDeviceToHost)"
assert translator.memcpy.parseString(v)[0].f_str(True,False,8).replace(" ","")  == "hipMemcpy(a,c_loc(b),1_8*(8)*(n),hipMemcpyHostToDevice)"
assert translator.memcpy.parseString(v)[0].f_str(False,False,8).replace(" ","") == "hipMemcpy(c_loc(a),c_loc(b),1_8*(8)*(n),hipMemcpyHostToHost)"

v = testdata[1]
assert translator.memcpy.parseString(v)[0].f_str(True,True,8).replace(" ","")   == "hipMemcpy(type0%a,type1%b,1_8*(8)*(n),hipMemcpyDeviceToDevice)"
assert translator.memcpy.parseString(v)[0].f_str(False,True,8).replace(" ","")  == "hipMemcpy(c_loc(type0%a),type1%b,1_8*(8)*(n),hipMemcpyDeviceToHost)"
assert translator.memcpy.parseString(v)[0].f_str(True,False,8).replace(" ","")  == "hipMemcpy(type0%a,c_loc(type1%b),1_8*(8)*(n),hipMemcpyHostToDevice)"
assert translator.memcpy.parseString(v)[0].f_str(False,False,8).replace(" ","") == "hipMemcpy(c_loc(type0%a),c_loc(type1%b),1_8*(8)*(n),hipMemcpyHostToHost)"
#print(translator.memcpy.parseString(v)[0].f_str(True,True,8).replace(" ","")  )
#print(translator.memcpy.parseString(v)[0].f_str(False,True,8).replace(" ","") )
#print(translator.memcpy.parseString(v)[0].f_str(True,False,8).replace(" ","") )
#print(translator.memcpy.parseString(v)[0].f_str(False,False,8).replace(" ",""))

print("SUCCESS")