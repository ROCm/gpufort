#!/usr/bin/env python3
import test
import translator.translator

testdata ="""
a0(1,2)
a0(1)
""".strip(" ").strip("\n").split("\n")

test.run(
   expression     = translator.funcCall,
   testdata       = testdata,
   tag            = "funcCall",
   raiseException = True
)

for v in testdata:
    print(translator.funcCall.parseString(v)[0].cStr())
