#!/usr/bin/env python3
import sys
import test
import translator.translator

for lib in ["FFTXlib","LAXlib","PW"]:
    testdata = open("testdata/use-{}.txt".format(lib),"r").readlines()
    test.run(
       expression     = translator.use,
       testdata       = testdata,
       tag            = "use-{}".format(lib),
       raiseException = True
    )
