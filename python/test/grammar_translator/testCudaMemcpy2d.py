#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import test
from gpufort import translator

translator.bounds.parseString("(1:n)")

testdata ="""
cudaMemcpy2D( array_out(d1_start, d2_start) , d1_ld, array_in(d1_start, d2_start), d2_ld, d1_size, d2_size )
cudaMemcpy2D(msg_dest_d,SIZE(msg_dest_d,1),msg_sour_d,SIZE(msg_sour_d,1),SIZE(msg_sour_d,1),SIZE(msg_sour_d,2),cudaMemcpyDeviceToDevice)
""".strip(" ").strip("\n").splitlines()

test.run(
   expression     = translator.cuf_cudamemcpy2D,
   testdata       = testdata,
   tag            = "cudaMemcpy2D",
   raiseException = True
)

v = testdata[0]
assert translator.cuf_cudamemcpy2D.parseString(v)[0].f_str(True,True,8).replace(" ","")   == "hipMemcpy2D(array_out(d1_start,d2_start),1_8*(d1_ld)*(8),array_in(d1_start,d2_start),1_8*(d2_ld)*(8),1_8*(d1_size)*(8),1_8*(d2_size)*(1),hipMemcpyDeviceToDevice)"
assert translator.cuf_cudamemcpy2D.parseString(v)[0].f_str(False,True,8).replace(" ","")  == "hipMemcpy2D(c_loc(array_out(d1_start,d2_start)),1_8*(d1_ld)*(8),array_in(d1_start,d2_start),1_8*(d2_ld)*(8),1_8*(d1_size)*(8),1_8*(d2_size)*(1),hipMemcpyDeviceToHost)"
assert translator.cuf_cudamemcpy2D.parseString(v)[0].f_str(True,False,8).replace(" ","")  == "hipMemcpy2D(array_out(d1_start,d2_start),1_8*(d1_ld)*(8),c_loc(array_in(d1_start,d2_start)),1_8*(d2_ld)*(8),1_8*(d1_size)*(8),1_8*(d2_size)*(1),hipMemcpyHostToDevice)"
assert translator.cuf_cudamemcpy2D.parseString(v)[0].f_str(False,False,8).replace(" ","") == "hipMemcpy2D(c_loc(array_out(d1_start,d2_start)),1_8*(d1_ld)*(8),c_loc(array_in(d1_start,d2_start)),1_8*(d2_ld)*(8),1_8*(d1_size)*(8),1_8*(d2_size)*(1),hipMemcpyHostToHost)"
#print(translator.cuf_cudamemcpy2D.parseString(v)[0].f_str(True,True,8).replace(" ","")  )
#print(translator.cuf_cudamemcpy2D.parseString(v)[0].f_str(False,True,8).replace(" ","") )
#print(translator.cuf_cudamemcpy2D.parseString(v)[0].f_str(True,False,8).replace(" ","") )
#print(translator.cuf_cudamemcpy2D.parseString(v)[0].f_str(False,False,8).replace(" ",""))
#
v = testdata[1]
assert translator.cuf_cudamemcpy2D.parseString(v)[0].f_str(True,True,8).replace(" ","")   == "hipMemcpy2D(msg_dest_d,1_8*(SIZE(msg_dest_d,1))*(8),msg_dest_d,1_8*(SIZE(msg_sour_d,1))*(8),1_8*(SIZE(msg_sour_d,1))*(8),1_8*(SIZE(msg_sour_d,2))*(1),cudaMemcpyDeviceToDevice)"
assert translator.cuf_cudamemcpy2D.parseString(v)[0].f_str(False,True,8).replace(" ","")  == "hipMemcpy2D(msg_dest_d,1_8*(SIZE(msg_dest_d,1))*(8),msg_dest_d,1_8*(SIZE(msg_sour_d,1))*(8),1_8*(SIZE(msg_sour_d,1))*(8),1_8*(SIZE(msg_sour_d,2))*(1),cudaMemcpyDeviceToDevice)"
assert translator.cuf_cudamemcpy2D.parseString(v)[0].f_str(True,False,8).replace(" ","")  == "hipMemcpy2D(msg_dest_d,1_8*(SIZE(msg_dest_d,1))*(8),msg_dest_d,1_8*(SIZE(msg_sour_d,1))*(8),1_8*(SIZE(msg_sour_d,1))*(8),1_8*(SIZE(msg_sour_d,2))*(1),cudaMemcpyDeviceToDevice)"
assert translator.cuf_cudamemcpy2D.parseString(v)[0].f_str(False,False,8).replace(" ","") == "hipMemcpy2D(msg_dest_d,1_8*(SIZE(msg_dest_d,1))*(8),msg_dest_d,1_8*(SIZE(msg_sour_d,1))*(8),1_8*(SIZE(msg_sour_d,1))*(8),1_8*(SIZE(msg_sour_d,2))*(1),cudaMemcpyDeviceToDevice)"
#print(translator.cuf_cudamemcpy2D.parseString(v)[0].f_str(True,True,8).replace(" ","")  )
#print(translator.cuf_cudamemcpy2D.parseString(v)[0].f_str(False,True,8).replace(" ","") )
#print(translator.cuf_cudamemcpy2D.parseString(v)[0].f_str(True,False,8).replace(" ","") )
#print(translator.cuf_cudamemcpy2D.parseString(v)[0].f_str(False,False,8).replace(" ",""))

print("SUCCESS")