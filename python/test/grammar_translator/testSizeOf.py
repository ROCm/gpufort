#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
import sys
import test
import translator.translator

testdata = []


test.run(
   expression     = translator.sizeOf,
   testdata       = testdata,
   tag            = "declaration-{}".format(lib),
   raiseException = True
)
