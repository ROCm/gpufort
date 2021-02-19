#!/usr/bin/env python3
import sys
import test
#import translator.translator

import grammar as translator

for lib in ["LAXlib","PW","FFTXlib"]:
    testdata = open("testdata/datatype-{}.txt".format(lib),"r").readlines() 

    test.run(
       expression     = translator.datatype,
       testdata       = testdata,
       tag            = "datatype-{}".format(lib),
       raiseException = True
    )
