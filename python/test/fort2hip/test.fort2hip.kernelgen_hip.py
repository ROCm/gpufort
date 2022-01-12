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
    def test_00_init_kernel_generator(self):
        pass
    def test_01_render_gpu_kernel_cpp(self):
        print("\n".join(self.kernelgen.render_gpu_kernel_cpp()))
    def test_02_render_begin_kernel_comment_cpp(self):
        print("\n".join(self.kernelgen.render_begin_kernel_comment_cpp()))
    def test_03_render_end_kernel_comment_cpp(self):
        print("\n".join(self.kernelgen.render_end_kernel_comment_cpp()))
    def test_04_render_begin_kernel_comment_f03(self):
        print("\n".join(self.kernelgen.render_begin_kernel_comment_f03()))
    def test_05_render_end_kernel_comment_f03(self):
        print("\n".join(self.kernelgen.render_end_kernel_comment_f03()))
    def test_06_render_hip_launcher_cpp(self):
        launcher = self.kernelgen.create_launcher_context(kind="hip",
                                                          debug_output=False,
                                                          used_modules=[])
        print("\n".join(self.kernelgen.render_gpu_launcher_cpp(launcher)))
    def test_07_render_hip_auto_launcher_cpp(self):
        launcher = self.kernelgen.create_launcher_context(kind="hip_auto",
                                                          debug_output=False,
                                                          used_modules=[],
                                                          problem_size=[1024,1,1],
                                                          grid  = [1,1,1],
                                                          block = [64,1,1])
        print("\n".join(self.kernelgen.render_gpu_launcher_cpp(launcher)))
    def test_08_render_hip_launcher_f03(self):
        launcher = self.kernelgen.create_launcher_context(kind="hip",
                                                          debug_output=False,
                                                          used_modules=[])
        print("\n".join(self.kernelgen.render_launcher_interface_f03(launcher)))
    def test_09_render_hip_auto_launcher_f03(self):
        launcher = self.kernelgen.create_launcher_context(kind="hip_auto",
                                                          debug_output=False,
                                                          used_modules=[],
                                                          problem_size=[1024,1,1],
                                                          grid  = [1,1,1],
                                                          block = [64,1,1])
        print("\n".join(self.kernelgen.render_launcher_interface_f03(launcher)))
    def test_10_render_cpu_launcher_cpp(self):
        launcher = self.kernelgen.create_launcher_context(kind="cpu",
                                                          debug_output=False,
                                                          used_modules=[])
        print("\n".join(self.kernelgen.render_cpu_launcher_cpp(launcher)))
    def test_11_render_cpu_launcher_f03(self):
        launcher = self.kernelgen.create_launcher_context(kind="cpu",
                                                          debug_output=False,
                                                          used_modules=[])
        print("\n".join(self.kernelgen.render_launcher_interface_f03(launcher)))
    def test_12_render_cpu_routine_f03(self):
        launcher = self.kernelgen.create_launcher_context(kind="cpu",
                                                          debug_output=False,
                                                          used_modules=[])
        print("\n".join(self.kernelgen.render_cpu_routine_f03(launcher)))

if __name__ == '__main__':
    unittest.main() 
