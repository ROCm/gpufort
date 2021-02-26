#!/usr/bin/env python3
import sys
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")
from grammar import *
from translator import *
testdata=[]
testdata.append("""!$acc kernels num_gangs(8)
          do j = 1,M
            do i= 1, N
                a(j,i) = 10
                b(j,i) = 5
            end do
          end do
        !$acc end kernels""")
for i in range(len(testdata)):
    cSnippet, KernelLaunchInfo, problemSize, identifierNames, loopVars, localLValues = convertAccConstruct(testdata[i])
    print(cSnippet)
    #print(KernelLaunchInfo)
    print(problemSize)
    print(identifierNames)
    print(loopVars)
    print(localLValues)
