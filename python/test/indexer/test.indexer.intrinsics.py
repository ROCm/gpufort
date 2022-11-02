#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import time
import unittest
import cProfile,pstats,io

import addtoplevelpath
from gpufort import indexer
from gpufort import util
from gpufort import linemapper

log_format = "[%(levelname)s]\tgpufort:%(message)s"
log_level                   = "debug"
util.logging.opts.verbose   = False
util.logging.opts.traceback = False
util.logging.init_logging("log.log",log_format,log_level)

preproc_options=""

PROFILING_ENABLE = False

index = []

class TestParseIntrinsics(unittest.TestCase):
    def setUp(self):
        global index
        self.index = index
        self.started_at = time.time()
        if PROFILING_ENABLE:
            profiler = cProfile.Profile()
            profiler.enable()
    def tearDown(self):
        elapsed = time.time() - self.started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 6)))
        if PROFILING_ENABLE:
            profiler.disable() 
            s = io.StringIO()
            sortby = 'cumulative'
            stats = pstats.Stats(profiler, stream=s).sort_stats(sortby)
            stats.print_stats(10)
            print(s.getvalue())
    def test_0_donothing(self):
        pass 
    def test_1_indexer_scan_files(self):
        global PROFILING_ENABLE
        global USE_EXTERNAL_PREPROCESSOR
    
        indexer.update_index_from_linemaps(
          linemapper.read_file(
            "gpufort_intrinsics.f90",
            preproc_options=preproc_options,
            include_dirs=["../../../include"]
          ),
          self.index
        )
    def test_2_indexer_write_module_files(self):
        indexer.write_gpufort_module_files(index,"./")
    def test_3_indexer_load_module_files(self):
        self.index.clear()
        indexer.load_gpufort_module_files(["./"],self.index)
      
if __name__ == '__main__':
    unittest.main() 
