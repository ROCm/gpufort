#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import sys
import test
from gpufort import translator


print(translator.dimension_value.copy().setParseAction(lambda tokens: "'{}'".format(translator.make_f_str(tokens[0]))).transformString(":,llb:lle"))
print(translator.dimension_value.copy().setParseAction(lambda tokens: "'{}'".format(translator.make_f_str(tokens[0]))).transformString("(:,llb:lle)"))
print(translator.dimension_value.copy().setParseAction(lambda tokens: "'{}'".format(translator.make_f_str(tokens[0]))).transformString("ue_gradivu_e(:,llb:lle)"))
print(translator.acc_present.parseString("present( ue_gradivu_e(:,llb:lle), Ai(:), ne(:,:), le(:), de(:))"))

testdata = []

test.run(
   expression     = (translator.complex_assignment | translator.matrix_assignment | translator.assignment),
   testdata       = testdata,
   tag            = None,
   raiseException = True
)