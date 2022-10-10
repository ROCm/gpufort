#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os,sys
import time
import unittest
import inspect
import pyparsing

import addtoplevelpath
from gpufort import translator

print("Running test '{}'".format(os.path.basename(__file__)),end="",file=sys.stderr)
        

testdata = [
  "s00",
  "s00 + ( s10 + s11) ",
  "s00 - ( s10 + s11*( s20 * s21 ) )",
  "s00 * ( s10, s11 )",
]        

class TestTraverseArithmeticExpression(unittest.TestCase):
    def setUp(self):
        global index
        self.started_at = time.time()
    def tearDown(self):
        elapsed = time.time() - self.started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 9)))
    def test_00_do_nothing(self):
        pass
    def test_01_count_rvalues(self):
        results = [1,3,5,3]
        for i,test in enumerate(testdata):
            num_rvalues = 0
            num_binary_ops = 0
            indent=""
            def visit_action_(ttnode,parents):
                nonlocal num_rvalues
                if isinstance(ttnode,translator.tree.TTRvalue):
                    num_rvalues += 1
                #if isinstance(ttnode,pyparsing.ParseResults):
                #    print(len([c for c in ttnode if isinstance(c,translator.tree.TTRvalue)]))
            print(test)
            arith_logic_expr = translator.tree.grammar.arith_logic_expr.parseString(test)[0]
            translator.tree.traversals.traverse(
                arith_logic_expr,
                visit_action_,
                translator.tree.traversals.no_action,
                translator.tree.traversals.no_crit)
            self.assertEqual(results[i],num_rvalues)
             
if __name__ == '__main__':
    unittest.main() 