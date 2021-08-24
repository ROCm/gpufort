#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
import test
import translator.translator

translator.matrix_ranges.parseString("(1:n)")

testdata ="""
if ( allocated(A_d) ) deallocate(A_d)
if ( .not. allocated(A_d) ) deallocate(A_d); allocate(A_h);
""".strip(" ").strip("\n").split("\n")

test.run(
   expression     = translator.singleLineIf,
   testdata       = testdata,
   tag            = "singleLineIf",
   raiseException = True
)
v = testdata[0]
print(translator.singleLineIf.transform_string(v))

v = testdata[1]
print(translator.singleLineIf.transform_string(v))

#assert translator.singleLineIf.parseString(v)[0].f_str(True,True,8).replace(" ","")   == "hipMemcpy(a,b,1_8*(8)*(n),hipMemcpyDeviceToDevice)"
#assert translator.singleLineIf.parseString(v)[0].f_str(False,True,8).replace(" ","")  == "hipMemcpy(c_loc(a),b,1_8*(8)*(n),hipMemcpyDeviceToHost)"
#assert translator.singleLineIf.parseString(v)[0].f_str(True,False,8).replace(" ","")  == "hipMemcpy(a,c_loc(b),1_8*(8)*(n),hipMemcpyHostToDevice)"
#assert translator.singleLineIf.parseString(v)[0].f_str(False,False,8).replace(" ","") == "hipMemcpy(c_loc(a),c_loc(b),1_8*(8)*(n),hipMemcpyHostToHost)"
#
#v = testdata[1]
#assert translator.singleLineIf.parseString(v)[0].f_str(True,True,8).replace(" ","")   == "hipMemcpy(type0%a,type1%b,1_8*(8)*(n),hipMemcpyDeviceToDevice)"
#assert translator.singleLineIf.parseString(v)[0].f_str(False,True,8).replace(" ","")  == "hipMemcpy(c_loc(type0%a),type1%b,1_8*(8)*(n),hipMemcpyDeviceToHost)"
#assert translator.singleLineIf.parseString(v)[0].f_str(True,False,8).replace(" ","")  == "hipMemcpy(type0%a,c_loc(type1%b),1_8*(8)*(n),hipMemcpyHostToDevice)"
#assert translator.singleLineIf.parseString(v)[0].f_str(False,False,8).replace(" ","") == "hipMemcpy(c_loc(type0%a),c_loc(type1%b),1_8*(8)*(n),hipMemcpyHostToHost)"
#print(translator.singleLineIf.parseString(v)[0].f_str(True,True,8).replace(" ","")  ) 
#print(translator.singleLineIf.parseString(v)[0].f_str(False,True,8).replace(" ","") )
#print(translator.singleLineIf.parseString(v)[0].f_str(True,False,8).replace(" ","") ) 
#print(translator.singleLineIf.parseString(v)[0].f_str(False,False,8).replace(" ",""))

print("SUCCESS")