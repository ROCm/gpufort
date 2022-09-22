#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os,sys
import time
import unittest
import addtoplevelpath
import inspect
from gpufort import translator

print("Running test '{}'".format(os.path.basename(__file__)),end="",file=sys.stderr)

class TestTraverseArithmeticExpression(unittest.TestCase):
    def setUp(self):
        global index
        self.started_at = time.time()
    def tearDown(self):
        elapsed = time.time() - self.started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 9)))
    def test_00_do_nothing(self):
        testdata = [
          "s00",
          "s00 + ( s10 + s11) ",
          "s00 + ( s10 + s11*( s20 * s21 ) )",
          "s00 + ( s10, s11 )",
        ]        
        for test in testdata:
            print("'"+test+"':")
            parse_result = translator.tree.grammar.arithmetic_expression.parseString(test)
            try:
                print(type(parse_result))  # class pyparsing.ParseResults
                print(type(parse_result[0])) #  class 'gpufort.translator.tree.fortran.TTArithmeticExpression'
                print(type(parse_result[0]._expr)) # class pyparsing.ParseResults
                print([type(c) for c in parse_result[0]._expr.asList()])
                print(type(parse_result[0]._expr[0])) # class 'gpufort.translator.tree.fortran.TTRValue'
                print(type(parse_result[0]._expr[1])) # class 'gpufort.translator.tree.fortran.TTOperator'
                if ( type(parse_result[0]._expr[2]) == translator.tree.TTComplexArithmeticExpression ):
                    print(type(parse_result[0]._expr[2]._real))
                    print(type(parse_result[0]._expr[2]._imag))
                print(type(parse_result[0]._expr[2]))
                print(type(parse_result[0]._expr[2][0]))
            except:
                pass
if __name__ == '__main__':
    unittest.main() 
