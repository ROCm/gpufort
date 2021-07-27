#!/usr/bin/env python3
import time
import unittest
import cProfile,pstats,io
import json

import addtoplevelpath
import linemapper.linemapper as linemapper
import utils.logging

LOG_FORMAT = "[%(levelname)s]\tgpufort:%(message)s"
utils.logging.VERBOSE    = False
utils.logging.initLogging("log.log",LOG_FORMAT,"warning")

PROFILING_ENABLE = False

index = []

class TestIndexer(unittest.TestCase):
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
        records                   = linemapper.readFile("test1.f90",options)
        result_lines              = linemapper.renderFile(records,stage="lines")
        result_raw_statements     = linemapper.renderFile(records,stage="raw_statements")
        result_statements         = linemapper.renderFile(records,stage="statements")
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
        #print(result_statements)
        #print(result_expandedStatements)

        def clean_(text):
            return text.replace(" ","").replace("\t","").replace("\n","").replace("\r","")

        self.assertEqual(clean_(result_lines),clean_(testdata_lines))
        self.assertEqual(clean_(result_raw_statements),clean_(testdata_raw_statements))
        self.assertEqual(clean_(result_statements),clean_(testdata_statements))
      
if __name__ == '__main__':
    unittest.main() 