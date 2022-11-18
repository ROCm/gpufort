#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os,sys
import time
import unittest
import addtoplevelpath
import inspect
import cProfile,pstats,io

PROFILING_ENABLE = False

from gpufort import translator
from gpufort import indexer

print("Running test '{}'".format(os.path.basename(__file__)),end="",file=sys.stderr)

scope = indexer.scope.create_scope_from_declaration_list(
"""
integer, parameter :: M = 10
integer, parameter :: N = 11
integer :: i,j,k
integer :: a(32,M,N), b(M,N), c(N)
real :: x,y,z,u
"""
)


testdata = [
  """\
!$acc kernels
c(:) = b(:,1)+3 
!$acc end kernels
""",
  """\
!$acc parallel loop reduction(+:j)
do i = 1, N
  j = j + 1
end do
!$acc end parallel
""",

  """\
!$acc parallel
!$acc loop gang(2)
do k = 1, N
  c(k) = 3
  !$acc loop worker(4)
  do j = 1, M
     b(j,k) = 2*c(k)
     !$acc loop vector(8) private(y)
     do i = -5,10,2
       a(i,j,k) = b(j,k) + c(k)
     end do

     label: do i = -5,10,2
       a(i,j,k) = b(j,k) + c(k)
       if ( i == 3 ) then 
          exit label
       endif
     end do label
    
     !$acc loop vector private(z)
     do i = -5,10,2
       a(i,j,k) = 2*a(i,j,k)
     end do
  end do
end do

!$acc loop gang worker vector dtype(radeon) tile(64) device_type(nvidia) tile(32) private(u)
do k = 1, M
  a(0,0,k) = 3
end do
!$acc end parallel
""",
]

class TestTransformAcc(unittest.TestCase):
    def setUp(self):
        global PROFILING_ENABLE
        self.started_at = time.time()
        if PROFILING_ENABLE:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
    def tearDown(self):
        global PROFILING_ENABLE
        elapsed = time.time() - self.started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 9)))
        if PROFILING_ENABLE:
            self.profiler.disable() 
            s = io.StringIO()
            sortby = 'cumulative'
            stats = pstats.Stats(self.profiler, stream=s).sort_stats(sortby)
            stats.print_stats(10)
            print(s.getvalue())
    def test_01_transform(self):
        device_type = "radeon"
        #for i,test in enumerate(testdata[-1:]):
        for i,test in enumerate(testdata[0:]):
            statements = test.splitlines()
            parse_result = translator.parser.parse_fortran_code(
              statements,result_name=None,scope=scope
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
