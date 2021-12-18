#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import addtoplevelpath
import os,sys
import time
import unittest
import cProfile,pstats,io
import re

import utils.logging
import translator.translator as translator
import indexer.indexerutils as indexerutils

log_format = "[%(levelname)s]\tgpufort:%(message)s"
log_level                   = "warning"
utils.logging.VERBOSE       = False
utils.logging.TRACEBACK = False
utils.logging.init_logging("log.log",log_format,log_level)

class TestParseLoopKernel(unittest.TestCase):
    def setUp(self):
        self.scope = indexerutils.create_scope_from_declaration_list(
        """
        integer :: i,j,k,n,c
        integer,dimension(:)     :: a,b
        integer,dimension(:,:)   :: e
        integer,dimension(:,:,:) :: f
        """)
        self.testdata = []
        self.testdata.append("""
        !$acc parallel
        !$acc loop 
        do i = 1, n
          a(i) = 3;
        end do
        !$acc end parallel
        """)
        self.testdata.append("""
        !$acc kernels loop copyin(b) copy(a(:)) reduction(+:c) async(1)
        do i = 1, n
          a(i) = b(i) + c;
          c = c + a(i)
        end do
        """)
        
        self.testdata.append("""
        !$acc parallel loop 
        do i = 1, n
          a(i) = 3;
        end do
        """)
        
        self.testdata.append("""
        !$acc kernels loop collapse(2)
        do i = 1, n
          do j = 1, n
            e(i,j) = 3;
          end do
        end do
        !$acc end kernels loop
        """)
        
        self.testdata.append("""
        !$acc parallel
        !$acc loop gang worker
        do i = 1, n
          !$acc loop vector
          do j = 1, n
            e(i,j) = 3;
          end do
        end do
        !$acc end parallel
        """)
        
        self.testdata.append("""
        !$acc kernels
        !$acc loop gang
        do k = 1, n
          !$acc loop worker
          do j = 1, n
            !$acc loop vector
            do i = 1, n
              f(i,j,k) = 3;
            end do
          end do
        end do
        !$acc end kernels
        """)
        # results
        self.results = []
        self.results.append("""
        !$omp target
        !$omp parallel do 
        do i = 1, n
          a(i) = 3;
        end do
        !$omp end target
        """)
        self.results.append("""
        !$omp target teams distribute parallel do reduction(+:c) map(to:b) &
        !$omp map(tofrom:a(:)) nowait depend(to:b) depend(tofrom:a)
        do i = 1, n
          a(i) = b(i) + c;
          c = c + a(i)
        end do
        """)
        self.results.append("""
        !$omp target teams distribute parallel do 
        do i = 1, n
          a(i) = 3;
        end do
        """)
        self.results.append("""
        !$omp target teams distribute parallel do
        do i = 1, n
          do j = 1, n
            e(i,j) = 3;
          end do
        end do
        
        """)
        self.results.append("""
        !$omp target
        !$omp teams distribute parallel do
        do i = 1, n
          !$omp do simd
          do j = 1, n
            e(i,j) = 3;
          end do
        end do
        !$omp end target
        """)
        self.results.append("""
        !$omp target
        !$omp teams distribute parallel do
        do k = 1, n
          !$omp do
          do j = 1, n
            !$omp do simd
            do i = 1, n
              f(i,j,k) = 3;
            end do
          end do
        end do
        !$omp end target
        """)
        self.started_at = time.time()
    def tearDown(self):
        elapsed = time.time() - self.started_at
        print('{} ({}s, {} kernels)'.format(self.id(), round(elapsed, 6),len(self.testdata)))
    def test_0_parse_loop_kernel(self):
        for snippet in self.testdata:
            try:
                result = translator.parse_loop_kernel(snippet.split("\n"),self.scope)
            except Exception as e:
                print("failed to parse '{}'".format(snippet),file=sys.stderr)
                raise e
    def test_1_translate_loop_kernel_to_omp(self):
        def strip_(text):
            return re.sub("\s+|\t+|\n+|&","",text)
        for i,snippet in enumerate(self.testdata):
            try:
                result = translator.parse_loop_kernel(snippet.split("\n"),self.scope)
                omp_result = result.omp_f_str(snippet)
                self.assertEqual(strip_(omp_result),strip_(self.results[i]))
            except Exception as e:
                print("failed to parse '{}'".format(snippet),file=sys.stderr)
                raise e

if __name__ == '__main__':
    unittest.main() 