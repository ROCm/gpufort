#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os,sys
import time
import unittest
import addtoplevelpath
from gpufort import grammar
from gpufort import translator

print("Running test '{}'".format(os.path.basename(__file__)),end="",file=sys.stderr)

class TestGrammarFunctionCall(unittest.TestCase):
    def setUp(self):
        global index
        self.started_at = time.time()
    def tearDown(self):
        elapsed = time.time() - self.started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 9)))
    def test_0(self):
        testdata = [
          "MIN(jte,jde-1,jdk)",
          "foo(jte)",
          "foo(jte,jde)",
          "foo(pcbywdth)",
          "MIN(jte,jde-1,pec_bdy_width)",
          "foo(spcbywdth)",
          "foo(spcbdywdth)",
          "foo(specbdywdth)",
          "foo(specbdywidth)",
          "foo(spec_bdy_width)",
          "foo(jte,jde,spec_bdy_width)",
          "foo(jte,jde-1,spec_bdy_width)",
          "MIN(jte,jde-1,spec_bdy_width)",
          "b(:,i:i+2)"
        ]
        for snippet in testdata:
            try:
                grammar.tensor_access.parseString(snippet,parseAll=True)
                translator.tree.grammar.tensor_access.parseString(snippet,parseAll=True)[0].fstr()
                #print(grammar.tensor_access.parseString(snippet,parseAll=True))
                #print(translator.tree.grammar.tensor_access.parseString(snippet,parseAll=True)[0].fstr())
                print("successfully parsed '{}'".format(snippet))
            except Exception as e:
                self.assertTrue(False, "failed to parse '{}'".format(snippet)) 

if __name__ == '__main__':
    unittest.main() 