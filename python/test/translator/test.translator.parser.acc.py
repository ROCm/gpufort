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

scope = None

#semantics = translator.semantics.Semantics()

class TestGrammarAcc(unittest.TestCase):
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

    def _assert_fails_to_parse(self,test):
        #print(test)
        passed = True
        try:
            ttaccdir = translator.parser.parse_acc_directive(test)
        except Exception as e: 
            #print(e)
            passed = False
            pass
        self.assertFalse(passed)

    def test_01_parse_acc_wait(self):
        testdata = [
          "!$acc wait",
          "!$acc wait(1,2,queue_id)",
          "!$acc wait(a*2*i)",
          "!$acc wait(a*2*i) async",
          "!$acc wait(1,2,queue_id,a*2*i) async(1,2,queue_id,a*2*i)",
          "!$acc wait(1,2,queue_id,a*2*i) async(1,2,queue_id,a*2*i) if(.true.)",
        ]
        for i,test in enumerate(testdata):
            #print(test)
            ttaccdir = translator.parser.parse_acc_directive(test)
            self.assertEqual(type(ttaccdir),translator.tree.TTAccWait)
            #TestGrammarAcc.parse_function_call_parse_results.append(ttaccdir)
    def test_02_parse_acc_data(self):
        # expected to fail
        testdata_fail = [
          "!$acc data", # not required clause present
          "!$acc data if(.true.) if(.true.)", # two if
        ]
        for i,test in enumerate(testdata_fail):
            self._assert_fails_to_parse(test)
        # expected to pass
        testdata = [
          "!$acc data copy(a)",
          "!$acc data default(none)",
          "!$acc data default(present)",
          "!$acc data copy(a) if (.true.)",
          "!$acc data copy(a) create(b) if(.true.)",
        ]
        for i,test in enumerate(testdata):
            #print(test)
            ttaccdir = translator.parser.parse_acc_directive(test)
            self.assertEqual(type(ttaccdir),translator.tree.TTAccData)

    def test_03_parse_acc_update(self):
        # expected to fail
        testdata_fail = [
          "!$acc update"
          "!$acc update device(3)",
        ]
        for i,test in enumerate(testdata_fail):
            self._assert_fails_to_parse(test)
        # expected to pass
        testdata = [
          "!$acc update self(a)",
          "!$acc update device(b)",
          "!$acc update self(a) device(b)",
          "!$acc update device(a,b,c,d) device(e)",
          "!$acc update device(a) device(b) async(2*i)",
        ]
        for i,test in enumerate(testdata):
            #print(test)
            ttaccdir = translator.parser.parse_acc_directive(test)
            self.assertEqual(type(ttaccdir),translator.tree.TTAccUpdate)
    
    def test_04_parse_acc_kernels(self):
        # expected to fail
        testdata_fail = [
          "!$acc kernels reduction(+:x)",
          "!$acc kernels gang",
          "!$acc kernels worker",
          "!$acc kernels vector",
          "!$acc kernels num_gangs",
          "!$acc kernels num_workers",
          "!$acc kernels vector_length",
          "!$acc kernels gang(10+1)",
          "!$acc kernels worker(10+1)",
          "!$acc kernels vector(10+1)",
        ]
        for i,test in enumerate(testdata_fail):
            self._assert_fails_to_parse(test)
        # expected to pass
        testdata = [
          "!$acc kernels",
          "!$acc kernels num_gangs(10+1) async(2*i)",
          "!$acc kernels num_workers(10+1) async(2*i)",
          "!$acc kernels vector_length(10+1) copy(x) copyin(z) async(2*i)",
        ]
        for i,test in enumerate(testdata):
            #print(test)
            ttaccdir = translator.parser.parse_acc_directive(test)
            self.assertEqual(type(ttaccdir),translator.tree.TTAccKernels)
    
    def test_05_parse_acc_kernels_loop(self):
        # expected to fail
        testdata_fail = [
        ]
        for i,test in enumerate(testdata_fail):
            self._assert_fails_to_parse(test)
        # expected to pass
        testdata = [
          "!$acc kernels loop",
          "!$acc kernels loop gang(10) worker vector",
          "!$acc kernels loop gang(10) worker vector_length(32)",
          "!$acc kernels loop gang(10) worker vector_length(32) reduction(min:x)",
        ]
        for i,test in enumerate(testdata):
            #print(test)
            ttaccdir = translator.parser.parse_acc_directive(test)
            self.assertEqual(type(ttaccdir),translator.tree.TTAccKernelsLoop)

if __name__ == '__main__':
    unittest.main() 
