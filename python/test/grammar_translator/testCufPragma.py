#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import sys
sys.path.append("..")
from translator import *

cufKernelDoVariants="""!$cuf kernel do
!$cuf kernel do
!$cuf kernel do (1)
!$cuf kernel do (1)
!$cuf kernel do (1) <<<*, *>>>
!$cuf kernel do (1) <<<*, *>>>
!$cuf kernel do (1) <<<*,*>>>
!$cuf kernel do(1)
!$cuf kernel do(1) <<<,>>>
!$cuf kernel do(1) <<<*, *>>>
!$cuf kernel do(1) <<<*,*>>>
!$cuf kernel do (1) <<<*,*,0,stream>>>
!$cuf kernel do (1) <<<*,*,0,dfft%a2a_comp>>>
!$cuf kernel do (1) <<<*,*,0,desc%stream_scatter_yz(1)>>>
!$cuf kernel do(1)<<<*,*,0,stream>>>
!$cuf kernel do(1) <<<*,*,0,stream>>>
!$cuf kernel do(1) <<<*,(16,16,1),0,stream>>>
!$cuf kernel do (2)
!$cuf kernel do(2)
!$cuf kernel do(2)<<<*,*>>>
!$cuf kernel do(2) <<<,>>>
!$cuf kernel do(2) <<<*,*>>>
!$cuf kernel do(2) <<<*,*,0,desc%stream_scatter_yz(iproc3)>>>
!$cuf kernel do(2) <<<*,*, 0, dfft%bstreams(batch_id)>>>
!$cuf kernel do (2) <<<*,*,0,stream>>>
!$cuf kernel do(2) <<<*,*,0,stream>>>
!$cuf kernel do(3)
!$cuf kernel do(3) <<<*,*>>>
!$cuf kernel do(3) <<<*,*, 0, dfft%a2a_comp>>>
!$cuf kernel do(3) <<<*,*,0,dfft%a2a_comp>>>
!$cuf kernel do(3) <<<*,(16,16,1), 0, stream>>>
!$cuf kernel do(4)""".splitlines()

for v in cufKernelDoVariants:
  try:
     print("{} -> {}".format(v,cufKernelDo.parseString(v)[0]))
  except Exception as e:
     print("failed to parse {}".format(v))
     raise e

print("SUCCESS!")