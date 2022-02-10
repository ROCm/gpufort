#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import sys
import test
import gpufort.translator
import addtoplevelpath
from gpufort import grammar

print(translator.kernel_launch_args.parseString("<<< grid , tBloc k >>>")[0])

testdata = """
CALL test_gpu<<<grid,tBlock>>>(A_d,5)
""".strip("\n").strip(" ").strip("\n").splitlines()

test.run(
   expression     = translator.cuf_kernel_call,
   testdata       = testdata,
   tag            = "cuf_kernel_call",
   raiseException = True
)