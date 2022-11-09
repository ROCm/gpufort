#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os,sys
import time
import unittest
import addtoplevelpath
import inspect
import cProfile,pstats,io

from gpufort import translator
from gpufort import indexer

print("Running test '{}'".format(os.path.basename(__file__)),end="",file=sys.stderr)

PROFILING_ENABLE = False

scope = indexer.scope.create_scope_from_declaration_list("""\
implicit none
integer, parameter :: m = 2, n = 3, k = 4
integer :: alpha,beta,gamma
integer :: i,j,k
integer :: A(m),B(m,n),C(m,n,k)
""")

testdata_arithexpr_scalar_rhs = [
  "1", 
  "A(i) + B(i,j)",
  "A(i) + B(i,j) - C(i,j,k)",
  "A(i) + B(i,j)*C(i,j,k)",
  "A(i) + B(i,j)*C(i,j,k) + A(k)",
  "C(i,j,k)*(A(i) + B(i,j)*C(i,j,k) + A(k))",
]       

class TestAnalysisArithExpr(unittest.TestCase):
    def setUp(self):
        global PROFILING_ENABLE
        self.started_at = time.time()
        if PROFILING_ENABLE:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
    def tearDown(self):
        elapsed = time.time() - self.started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 9)))
        if PROFILING_ENABLE:
            self.profiler.disable() 
            s = io.StringIO()
            sortby = 'cumulative'
            stats = pstats.Stats(self.profiler, stream=s).sort_stats(sortby)
            stats.print_stats(10)
            print(s.getvalue())
    def test_00_do_nothing(self):
        pass
    def test_01_parse_arithexpr(self):
        for i,test in enumerate(testdata_arithexpr_scalar_rhs):
            ttarithexpr = translator.tree.grammar.arith_expr.parseString(
              test,parseAll=True
            )[0]
            translator.analysis.semantics.resolve_arith_expr(ttarithexpr,scope)
            #for ttnode in ttarithexpr.walk_preorder():
            #    print(type(ttnode))
            #print("")
            #for ttnode in ttarithexpr.walk_postorder():
            #    print(type(ttnode))

if __name__ == '__main__':
    unittest.main() 
