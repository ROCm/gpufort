#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import os
import json
import unittest
import cProfile,pstats,io,time
import addtoplevelpath
import linemapper.linemapper as linemapper
import linemapper.linemapperutils as linemapperutils
import indexer.indexer as indexer
import indexer.indexerutils as indexerutils 
import translator.translator as translator
import utils.logging
import fort2hip.kernelgen_hip

LOG_FORMAT = "[%(levelname)s]\tgpufort:%(message)s"
utils.logging.VERBOSE    = False
utils.logging.init_logging("log.log",LOG_FORMAT,"warning")

PROFILING_ENABLE = False

declaration_list= """\
integer, parameter :: N = 1000
integer :: i
integer(4) :: x(N), y(N), y_exact(N)
"""

annotated_loop_nest = """\
!$acc parallel loop present(x,y)
do i = 1, N
  x(i) = 1
  y(i) = 2
end do
"""  

class TestHipKernelGenerator4LoopNest(unittest.TestCase):
    def prepare(self,text):
        return text.strip().split("\n")
    def clean(self,text):
        return text.replace(" ","").replace("\t","").replace("\n","").replace("\r","")
    def setUp(self):
        global PROFILING_ENABLE
        global declaration_list
        if PROFILING_ENABLE:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
        self.started_at = time.time()
        # 
        scope              = indexerutils.create_scope_from_declaration_list(declaration_list)
        linemaps           = linemapper.read_lines(annotated_loop_nest.split("\n"))
        fortran_statements = linemapperutils.get_statement_bodies(linemaps)
        ttloopnest         = translator.parse_loop_kernel(fortran_statements,
                                                          scope)
        #print(self.ttloopnest.c_str())
        self.kernelgen = fort2hip.kernelgen_hip.HipKernelGenerator4LoopNest("mykernel",
                                                                            "abcdefgh",
                                                                            ttloopnest,
                                                                            scope,
                                                                            "\n".join(fortran_statements))
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
    def test_0_init_kernel_generator(self):
        pass
    def test_0_init_kernel_generator(self):
        print("\n".join(self.kernelgen.render_gpu_kernel_cpp()))

if __name__ == '__main__':
    unittest.main() 
