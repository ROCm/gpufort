#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
import addtoplevelpath
import sys
import test
#import translator.translator

import translator.translator as translator

for lib in ["LAXlib","PW","FFTXlib"]:
    testdata = open("testdata/declaration-{}.txt".format(lib),"r").readlines() 
    testdata.insert(0,"REAL(DP)::A(LDA,N),D(N),E(N),V(LDV,N)")
    testdata.insert(0,"REAL(DP)::A(LDA,N)")
    testdata.insert(0,"REAL(DP)::A")
    testdata.insert(0,"CHARACTER(LEN=*),INTENT(IN)::calling_routine,message")
    testdata.insert(0,"CHARACTER(LEN=*)::calling_routine,message")
    testdata.insert(0,"CHARACTER(kind=*)::calling_routine")
    testdata.insert(0,"CHARACTER(kind=*,len=*)::calling_routine")

    test.run(
       expression     = translator.declaration,
       testdata       = testdata,
       tag            = "declaration-{}".format(lib),
       raiseException = True
    )
