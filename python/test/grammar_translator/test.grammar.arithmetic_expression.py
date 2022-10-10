#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os,sys
import time
import unittest
import addtoplevelpath
from gpufort import grammar

print("Running test '{}'".format(os.path.basename(__file__)),end="",file=sys.stderr)

class TestGrammarArithmeticExpression(unittest.TestCase):
    def setUp(self):
        global index
        self.started_at = time.time()
    def tearDown(self):
        elapsed = time.time() - self.started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 9)))
    def test_0(self):
        testdata = [
          "- ( +a - b*(-b +c ))",
          "- ( max ( m , n ) )",
          "MIN( jte )",
          "MIN( jte, jde-1 )",
          "MIN(jte,jde-1,spec_bdy_width)",
        ]
        for snippet in testdata:
            try:
                grammar.arith_logic_expr.parseString(snippet,parseAll=True)
            except Exception as e:
                self.assertTrue(False, "failed to parse '{}'".format(snippet)) 

if __name__ == '__main__':
    unittest.main() 
