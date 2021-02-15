#!/usr/bin/env python3
import sys
import test
#import translator.translator

import grammar as translator

for lib in ["LAXlib","PW","FFTXlib"]:
    testdata = open("testdata/datatypePlusArgList-{}.txt".format(lib),"r").readlines() 

    test.run(
       expression     = translator.datatypePlusArgList,
       testdata       = testdata,
       tag            = "datatypePlusArgList-{}".format(lib),
       raiseException = True
    )
