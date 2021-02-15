#!/usr/bin/env python3
import sys
import test
import translator.translator

testdata = []


test.run(
   expression     = translator.sizeOf,
   testdata       = testdata,
   tag            = "declaration-{}".format(lib),
   raiseException = True
)
