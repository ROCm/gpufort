#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
import sys
import test
import translator.translator
import grammar as translator

print(translator.kernelLaunchArgs.parseString("<<< grid , tBloc k >>>")[0])

testdata = """
CALL test_gpu<<<grid,tBlock>>>(A_d,5)
""".strip("\n").strip(" ").strip("\n").splitlines()

test.run(
   expression     = translator.cudaKernelCall,
   testdata       = testdata,
   tag            = "cudaKernelCall",
   raiseException = True
)
