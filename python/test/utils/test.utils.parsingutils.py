#!/usr/bin/env python3
import time
import unittest
import cProfile,pstats,io
import json

import addtoplevelpath
import utils.logging
import utils.parsingutils

LOG_FORMAT = "[%(levelname)s]\tgpufort:%(message)s"
utils.logging.VERBOSE    = False
utils.logging.init_logging("log.log",LOG_FORMAT,"warning")

PROFILING_ENABLE = False

index = []

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

class TestParsingUtils(unittest.TestCase):
    def prepare(self,text):
        return text.strip().split("\n")
    def clean(self,text):
        return text.replace(" ","").replace("\t","").replace("\n","").replace("\r","")
    def setUp(self):
        global PROFILING_ENABLE
        if PROFILING_ENABLE:
            self._profiler = cProfile.Profile()
            self._profiler.enable()
        self._started_at = time.time()
    def tearDown(self):
        global PROFILING_ENABLE
        if PROFILING_ENABLE:
            self._profiler.disable() 
            s = io.StringIO()
            sortby = 'cumulative'
            stats = pstats.Stats(self._profiler, stream=s).sort_stats(sortby)
            stats.print_stats(10)
            print(s.getvalue())
        elapsed = time.time() - self._started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 6)))
    def test_1_split_fortran_line(self):
        for line in self.prepare(testdata1):
            indent,stmt_or_dir,comment,trailing_ws =\
              utils.parsingutils.split_fortran_line(line)
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
        result = utils.parsingutils.relocate_inline_comments(\
                   testdata2.split("\n"))
        self.assertEqual(self.clean("\n".join(result)),self.clean(testdata2_result))
if __name__ == '__main__':
    unittest.main() 
