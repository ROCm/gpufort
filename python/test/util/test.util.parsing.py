#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import time
import unittest
import cProfile,pstats,io
import json

import addtoplevelpath
from gpufort import util

LOG_FORMAT = "[%(levelname)s]\tgpufort:%(message)s"
util.logging.opts.verbose    = False
util.logging.init_logging("log.log",LOG_FORMAT,"warning")

PROFILING_ENABLE = False

testdata1 = \
"""
! comment
  ! comment
stmt_or_dir ! comment
!$acc stmt_or_dir
*$acc stmt_or_dir
c$acc stmt_or_dir
C$acc stmt_or_dir
!$ acc stmt_or_dir
! $acc comment
  !a$acc comment
"""

# whitespace at begin is important to
# as [cC] in first column indicates a comment
# line in Fortran 77.
testdata2 = \
"""  call myroutine( & ! comment 1
        arg1,&
        ! comment 2


        arg2) ! comment 3
"""

testdata2_result = \
"""  call myroutine( &
        arg1,&


        arg2)
  ! comment 1
        ! comment 2
        ! comment 3
"""

testdata3="k () + a ( b, c(d)+e(f)) + g(h(i,j+a(k(),2)))"""

testdata3_result= {
    "a" :  [('a ( b, c(d)+e(f))', [' b', ' c(d)+e(f)']), ('a(k(),2)', ['k()', '2'])],
    "b" :  [],
    "c" :  [('c(d)', ['d'])],
    "d" :  [],
    "f" :  [],
    "g" :  [('g(h(i,j+a(k(),2)))', ['h(i,j+a(k(),2))'])],
    "h" :  [('h(i,j+a(k(),2))', ['i', 'j+a(k(),2)'])],
    "i" :  [],
    "j" :  [],
    "k" :  [('k ()', []), ('k()', [])],
    }


class TestParsingUtils(unittest.TestCase):
    def prepare(self,text):
        return text.strip().splitlines()
    def clean(self,text):
        return text.replace(" ","").replace("\t","").replace("\n","").replace("\r","")
    def setUp(self):
        global PROFILING_ENABLE
        if PROFILING_ENABLE:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
        self.started_at = time.time()
    def tearDown(self):
        global PROFILING_ENABLE
        if PROFILING_ENABLE:
            self.profiler.disable() 
            s = io.StringIO()
            sortby = 'cumulative'
            stats = pstats.Stats(self.profiler, stream=s).sort_stats(sortby)
            stats.print_stats(10)
            print(s.getvalue())
        elapsed = time.time() - self.started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 6)))
    def test_1_split_fortran_line(self):
        for line in self.prepare(testdata1):
            indent,stmt_or_dir,comment,trailing_ws =\
              util.parsing.split_fortran_line(line)
            if "stmt_or_dir" in line:
                self.assertTrue(len(stmt_or_dir))
            else:
                self.assertFalse(len(stmt_or_dir))
            if "comment" in line:
                self.assertTrue(len(comment))
            else:
                self.assertFalse(len(comment))
        # 
    def test_2_relocate_inline_comments(self):
        result = util.parsing.relocate_inline_comments(\
                   testdata2.splitlines())
        self.assertEqual(self.clean("\n".join(result)),self.clean(testdata2_result))
    def test_3_extract_function_calls(self):
        for c in ["a","b","c","d","f","g","h","i","j","k"]:
            result = util.parsing.extract_function_calls(testdata3,c)
            #print(result)
            self.assertEqual(result,testdata3_result[c])
    def test_4_parse_use_statement(self):
        statements = [
          "use mymod",
          "use mymod, only: var1",
          "use mymod, only: var1, var2",
          "use mymod, only: var1, var2=>var3",
        ]
        results = [
          ('mymod', {}),
          ('mymod', {'var1': 'var1'}),
          ('mymod', {'var1': 'var1', 'var2': 'var2'}),
          ('mymod', {'var1': 'var1', 'var2': 'var3'}),
        ]
        for i,stmt in enumerate(statements):
            #print(util.parsing.parse_use_statement(stmt))
            self.assertEqual(util.parsing.parse_use_statement(stmt),results[i])
if __name__ == '__main__':
    unittest.main() 
