#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import time
import unittest
import cProfile,pstats,io
import json

import addtoplevelpath
from gpufort import linemapper
from gpufort import util

LOG_FORMAT = "[%(levelname)s]\tgpufort:%(message)s"
util.logging.opts.verbose = False
util.logging.init_logging("log.log",LOG_FORMAT,"warning")

PROFILING_ENABLE = False

index = []

class TestLinemapper(unittest.TestCase):
    def clean(self,text):
        return text.replace(" ","").replace("\t","").replace("\n","").replace("\r","")
    def setUp(self):
        global PROFILING_ENABLE
        if PROFILING_ENABLE:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
        self.started_at = time.time()
        self.extra = ""
    def tearDown(self):
        global PROFILING_ENABLE
        if PROFILING_ENABLE:
            self.profiler.disable() 
            s = io.StringIO()
            sortby = 'cumulative'
            stats = pstats.Stats(self.profiler, stream=s).sort_stats(sortby)
            stats.print_stats(10)
            print(s.getvalue())
        elapsed = time.time() - self.started_at
        print('{} ({}s{})'.format(self.id(), round(elapsed, 6),self.extra))
    def test_0_donothing(self):
        pass 
    def test_1_macro_substitution(self):
        macro_stack = [
          ( "b", [], "5" ),
          ( "a", ["x"], "(5*x)" ),
        ]
        
        testdata_true = [
          "defined(a)",
          "!!defined(a)",
          "!defined(x)",
          "defined(a) && !defined(x) && a(b) > 4"
        ]
        testdata_false = [
          "!defined(a)",
          "!!defined(x)",
          "!defined(a) || defined(x) || a(5) < 1",
          "!(defined(a) && !defined(x) && a(b) > 4)"
        ]
        for text in testdata_true:
            condition = util.macros.evaluate_condition(text,macro_stack)
            self.assertTrue(condition)
        for text in testdata_false:
            condition = util.macros.evaluate_condition(text,macro_stack)
            self.assertFalse(condition)
        numTests = len(testdata_true) + len(testdata_false)
        self.extra = ", performed {} checks".format(numTests)
    def test_2_full_test(self):
        preproc_options = "-DCUDA -DCUDA2 -DCUDA4"
        linemaps                  = linemapper.read_file("test1.f90",preproc_options=preproc_options)
        result_lines              = linemapper.render_file(linemaps,stage="lines")
        result_raw_statements     = linemapper.render_file(linemaps,stage="raw_statements")
        result_statements         = linemapper.render_file(linemaps,stage="statements")
        testdata_lines            = """
        program main

        if ( 1 > 0 ) print *, size8(c)
        
        print *, "else"
        
        print *, B
        
        end program main
        """
        testdata_raw_statements = """
        program main
        
        if ( 1 > 0 ) then
          print *, size8(c)
        endif
        
        print *, "else"
        
        print *, B
        
        end program main
        """
        testdata_statements = """
        program main
        
        if ( 1 > 0 ) then
          print *, 8*(5)*2
        endif
        
        print *, "else"
        
        print *, 5*6
        
        end program main
        """
        #print(result_lines)
        #print(result_raw_statements)
        #print(result_statements)

        self.assertEqual(self.clean(result_lines),self.clean(testdata_lines))
        self.assertEqual(self.clean(result_raw_statements),self.clean(testdata_raw_statements))
        self.assertEqual(self.clean(result_statements),self.clean(testdata_statements))
    def test_3_expand_single_line_if(self):
        linemaps = linemapper.read_file("test2.f90",preproc_options="")
        result_statements = linemapper.render_file(linemaps,stage="statements")
        testdata_statements =\
"""
program main
if( tha(i,j,k).gt. 1.0 ) then
  thrad = -1.0/(12.0*3600.0)
endif
end program
"""    
        self.assertEqual(self.clean(result_statements),self.clean(testdata_statements))
    def test_4_collapse_multiline_acc_directive(self):
        linemaps = linemapper.read_file("test3.f90",preproc_options="")
        result_statements = linemapper.render_file(linemaps,stage="statements")
        testdata_statements =\
"""
program main
!$acc data copyin(a,b) copyout(c_gpu)
!$acc parallel loop collapse(2)   reduction(+:tmp)
do j=1,colsB
do i=1,rowsA
tmp = 0.0
!$acc loop vector reduction(+:tmp)
do k=1,rowsB
tmp = tmp  + a(i,k) * b(k,j)
enddo
c_gpu(i,j) = tmp
enddo
enddo
!$acc end parallel
!$acc end data
end program
"""
        self.assertEqual(self.clean(result_statements),self.clean(testdata_statements))
         
if __name__ == '__main__':
    unittest.main() 
