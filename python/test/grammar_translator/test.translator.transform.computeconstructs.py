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
integer, parameter :: M = 10
integer, parameter :: N = 11
integer :: i,j
integer :: a(32,M,N)
"""
)


testdata = [
  """\
!$acc parallel
!$acc loop gang(2) worker(4) collapse(2)
do k = 1, M
  do j = 1, N
     !$acc loop vector(8)
     do i = -5,10,2
       a(i,j,k) = 1
     end do
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
    def test_01_transform(self):
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
            print(trafo_result.hip_grid_and_block_as_str(
              "NUM_WORKERS",
              "warpSize",
              "operator_max", # type: str
              "operator_loop_len", # type: str
              "operator_div_round_up", # type: str
              translator.tree.traversals.make_fstr
            ))

if __name__ == '__main__':
    unittest.main() 
