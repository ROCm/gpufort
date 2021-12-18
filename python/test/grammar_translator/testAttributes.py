#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import addtoplevelpath
import sys
import test
import translator.translator as translator

for lib in ["PW"]:
    testdata = open("testdata/attributes-{}.txt".format(lib),"r").readlines()
    test.run(
       expression     = translator.attributes,
       testdata       = testdata,
       tag            = "attributes-{}".format(lib),
       raiseException = True
    )