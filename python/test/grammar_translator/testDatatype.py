#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import sys
import test
#import gpufort.translator

import addtoplevelpath
from gpufort import grammar

for lib in ["LAXlib","PW","FFTXlib"]:
    testdata = open("testdata/datatype-{}.txt".format(lib),"r").readlines() 

    test.run(
       expression     = translator.datatype,
       testdata       = testdata,
       tag            = "datatype-{}".format(lib),
       raiseException = True
    )