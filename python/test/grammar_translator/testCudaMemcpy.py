#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import test
from gpufort import translator

translator.bounds.parseString("(1:n)")

testdata ="""
cudaMemcpy(a,b,c)
cudaMemcpy(a%b%c,b%e%f,c*5)
cudaMemcpyAsync(a,b,c,stream)
cudaMemcpy(a,b,c,cudaMemcpyHostToDevice)
""".strip(" ").strip("\n").splitlines()

test.run(
   expression     = translator.cuf_cudamemcpy,
   testdata       = testdata,
   tag            = "cudaMemcpy",
   raiseException = True
)

v = testdata[0]
assert translator.cuf_cudamemcpy.parseString(v)[0].f_str(True,True,8).replace(" ","")   == "hipMemcpy(a,b,1_8*(c)*(8),hipMemcpyDeviceToDevice)"
assert translator.cuf_cudamemcpy.parseString(v)[0].f_str(False,True,8).replace(" ","")  == "hipMemcpy(c_loc(a),b,1_8*(c)*(8),hipMemcpyDeviceToHost)"
assert translator.cuf_cudamemcpy.parseString(v)[0].f_str(True,False,8).replace(" ","")  == "hipMemcpy(a,c_loc(b),1_8*(c)*(8),hipMemcpyHostToDevice)"
assert translator.cuf_cudamemcpy.parseString(v)[0].f_str(False,False,8).replace(" ","") == "hipMemcpy(c_loc(a),c_loc(b),1_8*(c)*(8),hipMemcpyHostToHost)"
#print(translator.cuf_cudamemcpy.parseString(v)[0].f_str(True,True,8).replace(" ","")  )
#print(translator.cuf_cudamemcpy.parseString(v)[0].f_str(False,True,8).replace(" ","") )
#print(translator.cuf_cudamemcpy.parseString(v)[0].f_str(True,False,8).replace(" ","") )
#print(translator.cuf_cudamemcpy.parseString(v)[0].f_str(False,False,8).replace(" ",""))
#
v = testdata[1]
assert translator.cuf_cudamemcpy.parseString(v)[0].f_str(True,True,8).replace(" ","")   == "hipMemcpy(a%b%c,b%e%f,1_8*((c*5))*(8),hipMemcpyDeviceToDevice)"
assert translator.cuf_cudamemcpy.parseString(v)[0].f_str(False,True,8).replace(" ","")  == "hipMemcpy(c_loc(a%b%c),b%e%f,1_8*((c*5))*(8),hipMemcpyDeviceToHost)"
assert translator.cuf_cudamemcpy.parseString(v)[0].f_str(True,False,8).replace(" ","")  == "hipMemcpy(a%b%c,c_loc(b%e%f),1_8*((c*5))*(8),hipMemcpyHostToDevice)"
assert translator.cuf_cudamemcpy.parseString(v)[0].f_str(False,False,8).replace(" ","") == "hipMemcpy(c_loc(a%b%c),c_loc(b%e%f),1_8*((c*5))*(8),hipMemcpyHostToHost)"
#print(translator.cuf_cudamemcpy.parseString(v)[0].f_str(True,True,8).replace(" ","")  )
#print(translator.cuf_cudamemcpy.parseString(v)[0].f_str(False,True,8).replace(" ","") )
#print(translator.cuf_cudamemcpy.parseString(v)[0].f_str(True,False,8).replace(" ","") )
#print(translator.cuf_cudamemcpy.parseString(v)[0].f_str(False,False,8).replace(" ",""))

v = testdata[2]
assert translator.cuf_cudamemcpy.parseString(v)[0].f_str(True,True,8).replace(" ","")   == "hipMemcpyAsync(a,b,1_8*(c)*(8),hipMemcpyDeviceToDevice,stream)"
assert translator.cuf_cudamemcpy.parseString(v)[0].f_str(False,True,8).replace(" ","")  == "hipMemcpyAsync(c_loc(a),b,1_8*(c)*(8),hipMemcpyDeviceToHost,stream)"
assert translator.cuf_cudamemcpy.parseString(v)[0].f_str(True,False,8).replace(" ","")  == "hipMemcpyAsync(a,c_loc(b),1_8*(c)*(8),hipMemcpyHostToDevice,stream)"
assert translator.cuf_cudamemcpy.parseString(v)[0].f_str(False,False,8).replace(" ","") == "hipMemcpyAsync(c_loc(a),c_loc(b),1_8*(c)*(8),hipMemcpyHostToHost,stream)"
#print(translator.cuf_cudamemcpy.parseString(v)[0].f_str(True,True,8).replace(" ","")  )
#print(translator.cuf_cudamemcpy.parseString(v)[0].f_str(False,True,8).replace(" ","") )
#print(translator.cuf_cudamemcpy.parseString(v)[0].f_str(True,False,8).replace(" ","") )
#print(translator.cuf_cudamemcpy.parseString(v)[0].f_str(False,False,8).replace(" ",""))

v = testdata[3]
assert translator.cuf_cudamemcpy.parseString(v)[0].f_str(True,True,8).replace(" ","")   == "hipMemcpy(a,c_loc(b),1_8*(c)*(8),cudaMemcpyHostToDevice)"
assert translator.cuf_cudamemcpy.parseString(v)[0].f_str(False,True,8).replace(" ","")  == "hipMemcpy(a,c_loc(b),1_8*(c)*(8),cudaMemcpyHostToDevice)"
assert translator.cuf_cudamemcpy.parseString(v)[0].f_str(True,False,8).replace(" ","")  == "hipMemcpy(a,c_loc(b),1_8*(c)*(8),cudaMemcpyHostToDevice)"
assert translator.cuf_cudamemcpy.parseString(v)[0].f_str(False,False,8).replace(" ","") == "hipMemcpy(a,c_loc(b),1_8*(c)*(8),cudaMemcpyHostToDevice)"
#print(translator.cuf_cudamemcpy.parseString(v)[0].f_str(True,True,8).replace(" ","")  )
#print(translator.cuf_cudamemcpy.parseString(v)[0].f_str(False,True,8).replace(" ","") )
#print(translator.cuf_cudamemcpy.parseString(v)[0].f_str(True,False,8).replace(" ","") )
#print(translator.cuf_cudamemcpy.parseString(v)[0].f_str(False,False,8).replace(" ",""))

print("SUCCESS")