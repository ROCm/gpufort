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
    testdata = open("testdata/datatypePlusArgList-{}.txt".format(lib),"r").readlines() 

    test.run(
       expression     = translator.datatypePlusArgList,
       testdata       = testdata,
       tag            = "datatypePlusArgList-{}".format(lib),
       raiseException = True
    )