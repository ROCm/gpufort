#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os,sys
import time
import unittest
import addtoplevelpath
import inspect

from gpufort import translator
from gpufort import indexer

print("Running test '{}'".format(os.path.basename(__file__)),end="",file=sys.stderr)

device_type = "radeon"

scope = indexer.scope.create_scope_from_declaration_list(
"""
integer, parameter :: N = 10
integer :: i,j
integer :: a(N,N)
"""
)


testdata = [
  """\
!$acc parallel
!$acc loop gang(2) worker(4) vector(8) collapse(2)
do i = 1, N
  do j = 1, N
     a(i,j) = 1
  end do
end do""",
]

class TestTransformAcc(unittest.TestCase):
    def setUp(self):
        global index
        self.started_at = time.time()
    def tearDown(self):
        elapsed = time.time() - self.started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 9)))
    def test_00_do_nothing(self):
        for i,test in enumerate(testdata):
            statements = test.splitlines()
            parse_result = translator.parser.parse_fortran_code(
              statements,result_name=None
            )
            trafo_result = translator.transform.computeconstructs.map_to_hip_cpp(
              parse_result.body[0],
              scope,
              device_type 
            )
            print(trafo_result.generated_code)

if __name__ == '__main__':
    unittest.main() 
