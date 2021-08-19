#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
import addtoplevelpath
import sys
import test
import translator.translator as translator


print(translator.dimensionValue.copy().setParseAction(lambda tokens: "'{}'".format(translator.make_f_str(tokens[0]))).transformString(":,llb:lle"))
print(translator.dimensionValue.copy().setParseAction(lambda tokens: "'{}'".format(translator.make_f_str(tokens[0]))).transformString("(:,llb:lle)"))
print(translator.dimensionValue.copy().setParseAction(lambda tokens: "'{}'".format(translator.make_f_str(tokens[0]))).transformString("ue_gradivu_e(:,llb:lle)"))
print(translator.acc_present.parseString("present( ue_gradivu_e(:,llb:lle), Ai(:), ne(:,:), le(:), de(:))"))

testdata = []

test.run(
   expression     = (translator.complexAssignment | translator.matrixAssignment | translator.assignment),
   testdata       = testdata,
   tag            = None,
   raiseException = True
)