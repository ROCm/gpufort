#!/usr/bin/env python3


import os,sys
import time
import unittest
import cProfile,pstats,io
import re

import addtoplevelpath
from gpufort import util
from gpufort import translator
from gpufort import indexer

log_format = "[%(levelname)s]\tgpufort:%(message)s"
log_level                   = "warning"
util.logging.opts.verbose       = False
util.logging.opts.traceback = False
util.logging.init_logging("log.log",log_format,log_level)

class TestParseLoopAnnotation(unittest.TestCase):
    def setUp(self):
        self.testdata = [
          "!$acc loop",
          "!$acc kernels loop create(b) copy(a(:)) reduction(+:c) async(1)",
          "!$acc parallel loop",
          "!$acc loop gang worker",
          "!$acc loop gang",
          "!$acc parallel loop copyin(moist(:,:,:,:))",
          "!$acc parallel loop copyin(moist(:,:,:,1))",
          "!$acc parallel loop copyin(moist(:,:,:,P_QV))",
          "!$acc parallel loop collapse(3) gang vector copy(grid%alt,grid%p,grid%p_hyd) copyin(grid%al,grid%alb,grid%pb,grid%t_1) copyin(moist(:,:,:,P_QV))",
        ]
        self.started_at = time.time()
    def tearDown(self):
        elapsed = time.time() - self.started_at
        print('{} ({}s, {} directives)'.format(self.id(), round(elapsed, 6),len(self.testdata)))
    def test_0_parse_loop_annotation(self):
        for snippet in self.testdata:
            try:
                result = translator.tree.grammar.loop_annotation.parseString(snippet,parseAll=True)
            except Exception as e:
                print("failed to parse '{}'".format(snippet),file=sys.stderr)
                raise e

if __name__ == '__main__':
    unittest.main() 
