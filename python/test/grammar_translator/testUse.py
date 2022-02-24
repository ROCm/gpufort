#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import sys
import test
from gpufort import translator

for lib in ["FFTXlib","LAXlib","PW"]:
    testdata = open("testdata/use-{}.txt".format(lib),"r").readlines()
    test.run(
       expression     = translator.use,
       testdata       = testdata,
       tag            = "use-{}".format(lib),
       raiseException = True
    )