#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os
import json
import jinja2
import unittest
import cProfile,pstats,io,time

import addtoplevelpath
from gpufort import util
from gpufort import fort2x

LOG_FORMAT = "[%(levelname)s]\tgpufort:%(message)s"
util.logging.opts.verbose    = False
util.logging.init_logging("log.log",LOG_FORMAT,"warning")

PROFILING_ENABLE = False

VERBOSE = False
def print_(text):
    global VERBOSE
    if VERBOSE:
        print(text)

max_rank = 3

class TestRenderGpufortTypes(unittest.TestCase):
    def prepare(self,text):
        return text.strip().splitlines()
    def clean(self,text):
        return text.replace(" ","").replace("\t","").replace("\n","").replace("\r","")
    def setUp(self):
        global PROFILING_ENABLE
        if PROFILING_ENABLE:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
        self.started_at = time.time()
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
        print('{} ({}s)'.format(self.id(), round(elapsed, 6)))
    def test_0_render_gpufort_array_ptr_header(self):
        print(fort2x.render.generate_code(
          "gpufort_array_ptr.template.h",
          context={
            "max_rank": max_rank
          }));
    def test_1_render_gpufort_array_ptr_source(self):
        print(fort2x.render.generate_code(
          "gpufort_array_ptr.template.cpp",
          context={
            "max_rank": max_rank
          }));

if __name__ == '__main__':
    unittest.main() 
