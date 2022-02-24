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
testdata.append("""!$acc kernels num_gangs(8)
          do j = 1,M
            do i= 1, N
                a(j,i) = 10
                b(j,i) = 5
            end do
          end do
        !$acc end kernels""")
for i in range(len(testdata)):
    c_snippet, KernelLaunchInfo, problem_size, identifier_names, loop_vars, localLValues = convertAccConstruct(testdata[i])
    print(c_snippet)
    #print(KernelLaunchInfo)
    print(problem_size)
    print(identifier_names)
    print(loop_vars)
    print(localLValues)