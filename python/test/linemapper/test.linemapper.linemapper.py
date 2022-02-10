#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import time
import unittest
import cProfile,pstats,io
import json

import __init__
from gpufort import linemapper
import util as util

LOG_FORMAT = "[%(levelname)s]\tgpufort:%(message)s"
util.logging.opts.verbose = False
util.logging.init_logging("log.log",LOG_FORMAT,"warning")

PROFILING_ENABLE = False

index = []

class TestLinemapper(unittest.TestCase):
    def __clean(self,text):
        return text.replace(" ","").replace("\t","").replace("\n","").replace("\r","")
    def setUp(self):
        global PROFILING_ENABLE
        if PROFILING_ENABLE:
            self._profiler = cProfile.Profile()
            self._profiler.enable()
        self._started_at = time.time()
    def tearDown(self):
        global PROFILING_ENABLE
        if PROFILING_ENABLE:
            self._profiler.disable() 
            s = io.StringIO()
            sortby = 'cumulative'
            stats = pstats.Stats(self._profiler, stream=s).sort_stats(sortby)
            stats.print_stats(10)
            print(s.getvalue())
        elapsed = time.time() - self._started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 6)))
    def test_0_donothing(self):
        pass 
    def test_1_full_test(self):
        options = "-DCUDA -DCUDA2"
        linemaps                  = linemapper.read_file("test1.f90",options)
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

        self.assertEqual(self.__clean(result_lines),self.__clean(testdata_lines))
        self.assertEqual(self.__clean(result_raw_statements),self.__clean(testdata_raw_statements))
        self.assertEqual(self.__clean(result_statements),self.__clean(testdata_statements))
    def test_2_expand_single_line_if(self):
        options = ""
        linemaps = linemapper.read_file("test2.f90",options)
        result_statements = linemapper.render_file(linemaps,stage="statements")
        testdata_statements =\
"""
program main
if( tha(i,j,k).gt. 1.0 ) then
  thrad = -1.0/(12.0*3600.0)
endif
end program
"""    
        self.assertEqual(self.__clean(result_statements),self.__clean(testdata_statements))
    def test_3_collapse_multiline_acc_directive(self):
        options = ""
        linemaps = linemapper.read_file("test3.f90",options)
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
        self.assertEqual(self.__clean(result_statements),self.__clean(testdata_statements))
      
if __name__ == '__main__':
    unittest.main() 