#!/usr/bin/env python3
import time
import unittest
import cProfile,pstats,io
import json

import addtoplevelpath
import preprocessor.preprocessor as preprocessor
import utils.logging

LOG_FORMAT = "[%(levelname)s]\tgpufort:%(message)s"
utils.logging.VERBOSE    = False
utils.logging.initLogging("log.log",LOG_FORMAT,"warning")

ENABLE_PROFILING = False

index = []

class TestIndexer(unittest.TestCase):
    def setUp(self):
        global ENABLE_PROFILING
        if ENABLE_PROFILING:
            profiler = cProfile.Profile()
            profiler.enable()
        self._started_at = time.time()
    def tearDown(self):
        global ENABLE_PROFILING
        if ENABLE_PROFILING:
            profiler.disable() 
            s = io.StringIO()
            sortby = 'cumulative'
            stats = pstats.Stats(profiler, stream=s).sort_stats(sortby)
            stats.print_stats(10)
            print(s.getvalue())
        elapsed = time.time() - self._started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 6)))
    def test_0_donothing(self):
        pass 
    def test_1_definitions(self):
        options = "-DCUDA -DCUDA2"
        result = preprocessor.preprocessAndNormalizeFortranFile("test1.f90",options)
        print(json.dumps(result,indent=2))
      
if __name__ == '__main__':
    unittest.main() 
