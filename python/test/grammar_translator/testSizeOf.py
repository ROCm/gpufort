#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import sys
import test
from gpufort import translator

testdata = []


test.run(
   expression     = translator.size_of,
   testdata       = testdata,
   tag            = "declaration-{}".format(lib),
   raiseException = True
)