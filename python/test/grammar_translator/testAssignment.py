#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import sys
import test
import gpufort.translator


for lib in ["PW","FFTXlib"]:
    testdata = open("testdata/assignment-{}.txt".format(lib),"r").readlines()
    test.run(
       expression     = (translator.matrix_assignment | translator.assignment),
       testdata       = testdata,
       tag            = "assignment-{}".format(lib),
       raiseException = False
    )