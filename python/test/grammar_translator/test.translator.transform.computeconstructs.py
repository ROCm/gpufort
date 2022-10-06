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

scope = indexer.scope.create_scope_from_declaration_list(
"""
integer, parameter :: M = 10
integer, parameter :: N = 11
integer :: i,j,k
integer :: a(32,M,N)
"""
)


testdata = [
  """\
!$acc parallel loop reduction(+:j)
do i = 1, N
  j = j + 1
end do
!$acc end parallel
""",
  """\
!$acc parallel
!$acc loop gang(2) worker(4) collapse(2) private(x)
do k = 1, M
  do j = 1, N
     x = 1.0
     !$acc loop vector(8) private(y)
     do i = -5,10,2
       a(i,j,k) = 1
     end do
    
     !$acc loop vector private(z)
     do i = -5,10,2
       a(i,j,k) = 2*a(i,j,k)
     end do
  end do
end do

!$acc loop gang worker vector dtype(radeon) tile(64) device_type(nvidia) tile(32) private(u)
do k = 1, M
  a(0,0,k) = 3
end do\
!$acc end parallel
""",
]

class TestTransformAcc(unittest.TestCase):
    def setUp(self):
        global index
        self.started_at = time.time()
    def tearDown(self):
        elapsed = time.time() - self.started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 9)))
    def test_01_transform(self):
        device_type = "radeon"
        for i,test in enumerate(testdata):
            statements = test.splitlines()
            parse_result = translator.parser.parse_fortran_code(
              statements,result_name=None
            )
            trafo_result = translator.transform.computeconstructs.map_to_hip_cpp(
              parse_result,
              scope,
              device_type 
            )
            print(trafo_result.generated_code)
            print(trafo_result.hip_grid_and_block_as_str(
              "NUM_WORKERS",
              "VECTOR_LENGTH",
              "operator_max", # type: str
              "operator_loop_len", # type: str
              "operator_div_round_up", # type: str
              translator.tree.traversals.make_fstr
            ))

if __name__ == '__main__':
    unittest.main() 
