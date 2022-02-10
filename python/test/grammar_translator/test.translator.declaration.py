#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os,sys
import time
import unittest
import addtoplevelpath
from gpufort import translator
from gpufort import util

print("Running test '{}'".format(os.path.basename(__file__)),end="",file=sys.stderr)

log_format = "[%(levelname)s]\tgpufort:%(message)s"
log_level                   = "debug2"
util.logging.opts.verbose       = False
util.logging.opts.traceback     = False
util.logging.init_logging("log.log",log_format,log_level)

testdata1_types = """
real
real(c_float)
real(8)
real*8
integer
logical
"""

testdata1_pass = """
{type} test
{type} test(:)
{type} test(:,:,:)
{type} test(5)
{type} test(5,6)
{type} test(n)
{type} test(n,k,c)
{type}                  :: test
{type},parameter        :: test
{type},intent(in)       :: test
{type},dimension(:)     :: test
{type}                  :: test(:)
{type},dimension(:,:,:) :: test
{type}                  :: test(:,:,:)
{type},pointer          :: test => null()
{type},pointer          :: test(:) => null()
{type},pointer          :: test(:,:,:) => null()
"""
testdata1_fail = """
{type},parameter     test
{type},intent(in)    test
"""

testdata2_pass = """
logical                  test = .false.
logical                  :: test = .false.
logical,parameter        :: test = .true.
logical,save,parameter   :: test = .false.
"""

class TestDeclaration(unittest.TestCase):
    def setUp(self):
        global index
        self.started_at = time.time()
    def tearDown(self):
        elapsed = time.time() - self.started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 9)))
    def prepare(self,testdata):
        return testdata.strip().splitlines()
    def test1_pass(self):
        testdata = self.prepare(testdata1_pass)
        types    = self.prepare(testdata1_types)
        for statement in testdata:
            for typ in types:
                translator.parse_declaration(statement.format(type=typ))
    def test1_fail(self):
        testdata = self.prepare(testdata1_pass)
        types    = self.prepare(testdata1_types)
        for statement in testdata:
            for typ in types:
                try:
                    translator.parse_declaration(statement.format(type=typ))
                    self.assertTrue(False)
                except Exception as e:
                    return e
    def test2_logicals_pass(self):
        testdata = self.prepare(testdata2_pass)
        for statement in testdata:
            translator.parse_declaration(statement)
if __name__ == '__main__':
    unittest.main() 