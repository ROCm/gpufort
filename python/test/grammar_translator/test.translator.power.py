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

print("Running test '{}'".format(os.path.basename(__file__)),end="",file=sys.stderr)

class TestTranslatorPower(unittest.TestCase):
    def setUp(self):
        global index
        self.started_at = time.time()
    def tearDown(self):
        elapsed = time.time() - self.started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 9)))
    def test_0_type_start_pass(self):
        testdata = [
          "a**0.5",
          "(t00/a)**2 >0.and.x<3",
          "EXP(3) **0.5 + 1.0e-03 + 2.0e+03",
          "p_surf = p00 * EXP ( -t00/a + ( (t00/a)**2 - 2.*g*grid%ht(i,j)/a/r_d ) **0.5 )",
          "vt(i,k,j) = 36.34*(qrr**0.1364) * vtden",
          "vt ( i , k , j ) = 36.34 * ( qrr ** 0.1364 ) * vtden",
        ]
        for snippet in testdata:
            criterion = True
            result = snippet
            #print(translator.prepostprocess.preprocess_fortran_statement(result))
            #print(translator.tree.grammar.power.parseString(result,parseAll=True))
            #while criterion:
            #    old_result = result
            #    result = translator.tree.grammar.power.transformString(result)
            #    criterion = old_result != result
            #print(result)

if __name__ == '__main__':
    unittest.main() 
