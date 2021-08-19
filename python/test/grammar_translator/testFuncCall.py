#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
import test
import translator.translator

testdata ="""
a0(1,2)
a0(1)
""".strip(" ").strip("\n").split("\n")

test.run(
   expression     = translator.funcCall,
   testdata       = testdata,
   tag            = "funcCall",
   raiseException = True
)

for v in testdata:
    print(translator.funcCall.parseString(v)[0].c_str())