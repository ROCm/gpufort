#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import addtoplevelpath
import os,sys
import time
import unittest
import grammar.grammar as grammar

print("Running test '{}'".format(os.path.basename(__file__)),end="",file=sys.stderr)

class TestIndexer(unittest.TestCase):
    def setUp(self):
        global index
        self._started_at = time.time()
    def tearDown(self):
        elapsed = time.time() - self._started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 9)))
    def test_0_type_start_pass(self):
        testdata = []
        testdata.append("type mytype")
        testdata.append("type :: mytype")
        testdata.append("type, bind(c) :: mytype")
        for snippet in testdata:
            try:
                grammar.type_start.parseString(snippet)
            except Exception as e:
                self.assertTrue(False, "failed to parse '{}'".format(snippet)) 
    def test_1_type_start_fail(self):
        testdata = []
        testdata.append("type(mytype) myvar")
        testdata.append("type(mytype) :: myvar")
        for snippet in testdata:
            try:
                grammar.type_start.parseString(snippet)
                self.assertTrue(False, "incorrectly parsed '{}'".format(snippet)) 
            except Exception as e:
                pass
    def test_2_type_end_pass(self):
        testdata = []
        testdata.append("end type")
        testdata.append("endtype")
        testdata.append("end type mytype")
        testdata.append("endtype mytype")
        for snippet in testdata:
            try:
                grammar.type_end.parseString(snippet)
            except Exception as e:
                self.assertTrue(False, "failed to parse '{}'".format(snippet)) 

if __name__ == '__main__':
    unittest.main() 