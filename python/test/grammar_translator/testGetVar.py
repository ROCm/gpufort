#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import sys
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")
from grammar import *
from translator import *
testdata=[]
testdata.append("""!$acc kernels num_gangs(8) copyin(a) copyout(b) create(c) copy(d)
            !$acc loop independent
          do j = 1,M
            do i= 1, N
                d(j,i) = c(j-1,i-1)
                a(j,i) = 10
                b(j,i) = 5
                c(j,i) = i+j
            end do
          end do
        !$acc end kernels""")
for i in range(len(testdata)):
    results = accConstruct.parseString(testdata[i])[0]
    print("copyin:",results.getInputVars())
    print("copyout:",results.getOutputVars())
    print("create: ",results.getMallocVars())
    #print(str(i)+".",results)
    #results.printTokens()
    #print(results.c_str())