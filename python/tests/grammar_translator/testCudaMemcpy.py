#!/usr/bin/env python3
import test
import translator.translator

translator.bounds.parseString("(1:n)")

testdata ="""
cudaMemcpy(a,b,c)
cudaMemcpy(a%b%c,b%e%f,c*5)
cudaMemcpyAsync(a,b,c,stream)
cudaMemcpy(a,b,c,cudaMemcpyHostToDevice)
""".strip(" ").strip("\n").split("\n")

test.run(
   expression     = translator.cufCudaMemcpy,
   testdata       = testdata,
   tag            = "cudaMemcpy",
   raiseException = True
)

v = testdata[0]
assert translator.cufCudaMemcpy.parseString(v)[0].fStr(True,True,8).replace(" ","")   == "hipMemcpy(a,b,1_8*(c)*(8),hipMemcpyDeviceToDevice)"
assert translator.cufCudaMemcpy.parseString(v)[0].fStr(False,True,8).replace(" ","")  == "hipMemcpy(c_loc(a),b,1_8*(c)*(8),hipMemcpyDeviceToHost)"
assert translator.cufCudaMemcpy.parseString(v)[0].fStr(True,False,8).replace(" ","")  == "hipMemcpy(a,c_loc(b),1_8*(c)*(8),hipMemcpyHostToDevice)"
assert translator.cufCudaMemcpy.parseString(v)[0].fStr(False,False,8).replace(" ","") == "hipMemcpy(c_loc(a),c_loc(b),1_8*(c)*(8),hipMemcpyHostToHost)"
#print(translator.cufCudaMemcpy.parseString(v)[0].fStr(True,True,8).replace(" ","")  ) 
#print(translator.cufCudaMemcpy.parseString(v)[0].fStr(False,True,8).replace(" ","") )
#print(translator.cufCudaMemcpy.parseString(v)[0].fStr(True,False,8).replace(" ","") ) 
#print(translator.cufCudaMemcpy.parseString(v)[0].fStr(False,False,8).replace(" ",""))
#
v = testdata[1]
assert translator.cufCudaMemcpy.parseString(v)[0].fStr(True,True,8).replace(" ","")   == "hipMemcpy(a%b%c,b%e%f,1_8*((c*5))*(8),hipMemcpyDeviceToDevice)"
assert translator.cufCudaMemcpy.parseString(v)[0].fStr(False,True,8).replace(" ","")  == "hipMemcpy(c_loc(a%b%c),b%e%f,1_8*((c*5))*(8),hipMemcpyDeviceToHost)"
assert translator.cufCudaMemcpy.parseString(v)[0].fStr(True,False,8).replace(" ","")  == "hipMemcpy(a%b%c,c_loc(b%e%f),1_8*((c*5))*(8),hipMemcpyHostToDevice)"
assert translator.cufCudaMemcpy.parseString(v)[0].fStr(False,False,8).replace(" ","") == "hipMemcpy(c_loc(a%b%c),c_loc(b%e%f),1_8*((c*5))*(8),hipMemcpyHostToHost)"
#print(translator.cufCudaMemcpy.parseString(v)[0].fStr(True,True,8).replace(" ","")  ) 
#print(translator.cufCudaMemcpy.parseString(v)[0].fStr(False,True,8).replace(" ","") )
#print(translator.cufCudaMemcpy.parseString(v)[0].fStr(True,False,8).replace(" ","") ) 
#print(translator.cufCudaMemcpy.parseString(v)[0].fStr(False,False,8).replace(" ",""))

v = testdata[2]
assert translator.cufCudaMemcpy.parseString(v)[0].fStr(True,True,8).replace(" ","")   == "hipMemcpyAsync(a,b,1_8*(c)*(8),hipMemcpyDeviceToDevice,stream)"
assert translator.cufCudaMemcpy.parseString(v)[0].fStr(False,True,8).replace(" ","")  == "hipMemcpyAsync(c_loc(a),b,1_8*(c)*(8),hipMemcpyDeviceToHost,stream)"
assert translator.cufCudaMemcpy.parseString(v)[0].fStr(True,False,8).replace(" ","")  == "hipMemcpyAsync(a,c_loc(b),1_8*(c)*(8),hipMemcpyHostToDevice,stream)"
assert translator.cufCudaMemcpy.parseString(v)[0].fStr(False,False,8).replace(" ","") == "hipMemcpyAsync(c_loc(a),c_loc(b),1_8*(c)*(8),hipMemcpyHostToHost,stream)"
#print(translator.cufCudaMemcpy.parseString(v)[0].fStr(True,True,8).replace(" ","")  ) 
#print(translator.cufCudaMemcpy.parseString(v)[0].fStr(False,True,8).replace(" ","") )
#print(translator.cufCudaMemcpy.parseString(v)[0].fStr(True,False,8).replace(" ","") ) 
#print(translator.cufCudaMemcpy.parseString(v)[0].fStr(False,False,8).replace(" ",""))

v = testdata[3]
assert translator.cufCudaMemcpy.parseString(v)[0].fStr(True,True,8).replace(" ","")   == "hipMemcpy(a,c_loc(b),1_8*(c)*(8),cudaMemcpyHostToDevice)"
assert translator.cufCudaMemcpy.parseString(v)[0].fStr(False,True,8).replace(" ","")  == "hipMemcpy(a,c_loc(b),1_8*(c)*(8),cudaMemcpyHostToDevice)"
assert translator.cufCudaMemcpy.parseString(v)[0].fStr(True,False,8).replace(" ","")  == "hipMemcpy(a,c_loc(b),1_8*(c)*(8),cudaMemcpyHostToDevice)"
assert translator.cufCudaMemcpy.parseString(v)[0].fStr(False,False,8).replace(" ","") == "hipMemcpy(a,c_loc(b),1_8*(c)*(8),cudaMemcpyHostToDevice)"
#print(translator.cufCudaMemcpy.parseString(v)[0].fStr(True,True,8).replace(" ","")  ) 
#print(translator.cufCudaMemcpy.parseString(v)[0].fStr(False,True,8).replace(" ","") )
#print(translator.cufCudaMemcpy.parseString(v)[0].fStr(True,False,8).replace(" ","") ) 
#print(translator.cufCudaMemcpy.parseString(v)[0].fStr(False,False,8).replace(" ",""))

print("SUCCESS")
