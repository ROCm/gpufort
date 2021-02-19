#!/usr/bin/env python3
import sys
import test
import translator.translator


for lib in ["PW","FFTXlib"]:
    testdata = open("testdata/assignment-{}.txt".format(lib),"r").readlines()
    test.run(
       expression     = (translator.matrixAssignment | translator.assignment),
       testdata       = testdata,
       tag            = "assignment-{}".format(lib),
       raiseException = False
    )
