#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os
import json
import unittest
import cProfile,pstats,io,time

import addtoplevelpath
from gpufort import util
from gpufort import fort2x
from gpufort import linemapper

LOG_FORMAT = "[%(levelname)s]\tgpufort:%(message)s"
util.logging.opts.verbose    = False
util.logging.init_logging("log.log",LOG_FORMAT,"warning")

PROFILING_ENABLE = False

VERBOSE = False
def print_(text):
    global VERBOSE
    if VERBOSE:
        print(text)

file_content= """\
module mymod
  type inner
    real(8)            :: scalar_double
    integer(4),pointer :: array_integer(:,:)
  end type inner

  type outer
    type(inner)                            :: single_inner
    type(inner),allocatable,dimension(:,:) :: array_inner
    integer(4),pointer                     :: array_integer(:,:,:)
  end type outer
end module
"""

class TestHipCodeGenerator(unittest.TestCase):
    def prepare(self,text):
        return text.strip().splitlines()
    def clean(self,text):
        return text.replace(" ","").replace("\t","").replace("\n","").replace("\r","")
    def setUp(self):
        global PROFILING_ENABLE
        global declaration_list
        if PROFILING_ENABLE:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
        self.started_at = time.time()

        self.codegen, self.linemaps = fort2x.hip.create_code_generator(file_content=file_content)
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
    def test_0_create_code_generator(self):
        pass
    def test_1_run_code_generator(self):
        self.codegen.run()
    def test_2_modify_file(self):
        global file_content
        print_(linemapper.modify_file(self.linemaps,
               file_content=file_content,
               ifdef_macro=None))
               #ifdef_macro="_GPUFORT"))
    def test_3_generate_cpp_files(self):
        print_(self.codegen.cpp_filegen.generate_code())
        for path,filegen in self.codegen.cpp_filegens_per_module:
            print_(filegen.generate_code())

if __name__ == '__main__':
    unittest.main() 
