# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from pyparsing import *
import sys

# TODO add siwtch for verbose output
def run(expression,testdata,tag,raiseException=False):
    successfullyParsed = []
    failedToParse = []
    
    for v in testdata:
        if v.strip()[0]!="#":
            try:
               #print("{} -> {}".format(v,declaration.parseString(v)[0]))
               #print("trying to parse: {}".format(v))
               result=expression.parseString(v)
               successfullyParsed.append(v)
               print("successfully parsed {0} -> {1}".format(v,str(result)))
            except Exception as e:
               print("failed to parse '{}'".format(v))
               failedToParse.append(v) 
               if raiseException:
                   print("Error in line {}: '{}'".format(e._intrnl_getattr___("lineno"),e._intrnl_getattr___("line"))) 
                   #if isinstance(e,ParseException):
                   raise e
    if not tag is None: 
        with open("testoutput/{}-success.txt".format(tag),"w") as output:
             output.writelines(successfullyParsed)
        with open("testoutput/{}-fail.txt".format(tag),"w") as output:
            output.writelines(failedToParse)
    if not len(failedToParse):
        print("PASSED")
    else:
        print("Summary: Failed to parse {0} of {1} test inputs".format(len(failedToParse),len(failedToParse)+len(successfullyParsed)))
        print("FAILED")