#!/usr/bin/env python3

import os
import json
import unittest
import cProfile,pstats,io,time

import addtoplevelpath
from gpufort import indexer
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

declaration_list= \
"""
type basic
  real(8)            :: scalar_double
  integer(4),pointer :: array_integer(:,:)
end type basic

type nested
  type(basic)                            :: single_basic
  type(basic),allocatable,dimension(:,:) :: array_basic
  integer(4),pointer                     :: array_integer(:,:,:)
end type nested

real(8)            :: scalar_double
integer(4),pointer :: array_integer(:,:,:)
type(nested)       :: derived_type
!
real(8)        :: scalar_double_local
real(8),shared :: scalar_double_shared
real(8) :: scalar_double_reduced
"""

class TestRenderKernel(unittest.TestCase):
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
        self.scope = indexer.scope.create_scope_from_declaration_list(declaration_list)
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
    def create_kernel_context(self,kernel_name,no_reduced_var=True):
        local_vars          = [ivar for ivar in self.scope["variables"] if "_local" in ivar["name"]]
        shared_vars         = [ivar for ivar in self.scope["variables"] if "_shared" in ivar["name"]]
        global_reduced_vars = [ivar for ivar in self.scope["variables"] if "_reduced" in ivar["name"]]
        global_vars         = [ivar for ivar in self.scope["variables"] if ivar not in local_vars and\
                              ivar not in shared_vars and ivar not in global_reduced_vars]
        if no_reduced_var:
            global_reduced_vars.clear() 
        else:
            for rvar in global_reduced_vars:
                rvar["op"] = "sum"
        #print([ivar["name"] for ivar in local_vars])
        #print([ivar["name"] for ivar in shared_vars])
        #print([ivar["name"] for ivar in global_vars])
        kernel = {}
        kernel["launch_bounds"]       = "__launch_bounds___(1024,1)"
        kernel["name"]                = kernel_name
        kernel["shared_vars"]         = shared_vars
        kernel["local_vars"]          = local_vars
        kernel["global_vars"]         = global_vars
        kernel["global_reduced_vars"] = global_reduced_vars
        kernel["c_body"]              = "// do nothing" 
        kernel["f_body"]              = "! do nothing"
        return kernel
    def create_hip_launcher_context(self,
                                    kernel_name,kind="hip_ps",
                                    debug_code=True):
        launcher = {}
        # for launcher interface
        launcher["kind"]         = kind
        if len(kind):
            launcher["name"]     = "_".join(["launch",kernel_name,kind])
        else:
            launcher["name"]     = "_".join(["launch",kernel_name])
        launcher["debug_code"] = True
        launcher["used_modules"]  = []
        return launcher
    def create_cpu_launcher_context(self,
                                           kernel_name):
        # for launcher interface
        launcher = {}
        launcher["name"] = "_".join(["launch",kernel_name,"cpu"])
        launcher["debug_code"]  = True
        launcher["kind"]          = "cpu"
        launcher["used_modules"]  = []
        return launcher
    def test_1_render_hip_kernel_cpp(self):
        print_(fort2x.hip.render.render_hip_kernel_cpp(self.create_kernel_context("mykernel")))
    def test_2_render_hip_kernel_with_reduced_vars_cpp(self):
        print_(fort2x.hip.render.render_hip_kernel_cpp(self.create_kernel_context("mykernel",False)))
    def test_3_render_hip_launcher_cpp(self):
        print_(fort2x.hip.render.render_hip_launcher_cpp(self.create_kernel_context("mykernel",False),
                                                        self.create_hip_launcher_context("mykernel")))
    def test_4_render_hip_launcher_with_reduced_vars_cpp(self):
        print_(fort2x.hip.render.render_hip_launcher_cpp(self.create_kernel_context("mykernel",False),
                                                        self.create_hip_launcher_context("mykernel")))
    def test_5_render_cpu_launcher_cpp(self):
        print_(fort2x.hip.render.render_cpu_launcher_cpp(self.create_kernel_context("mykernel"),
                                                        self.create_cpu_launcher_context("mykernel")))
    def test_6_render_cpu_launcher_with_reduced_vars_cpp(self):
        print_(fort2x.hip.render.render_cpu_launcher_cpp(self.create_kernel_context("mykernel",False),
                                                        self.create_cpu_launcher_context("mykernel")))
    def test_7_render_hip_launcher_f03(self):
        print_(fort2x.hip.render.render_launcher_f03(self.create_kernel_context("mykernel"),
                                                    self.create_hip_launcher_context("mykernel")))
    def test_8_render_hip_launcher_with_reduced_vars_f03(self):
        print_(fort2x.hip.render.render_launcher_f03(self.create_kernel_context("mykernel",False),
                                                    self.create_hip_launcher_context("mykernel")))
    def test_9_render_cpu_launcher_f03(self):
        print_(fort2x.hip.render.render_launcher_f03(self.create_kernel_context("mykernel"),
                                                    self.create_cpu_launcher_context("mykernel")))
    def test_10_render_cpu_launcher_with_reduced_vars_f03(self):
        print_(fort2x.hip.render.render_launcher_f03(self.create_kernel_context("mykernel",False),
                                                    self.create_cpu_launcher_context("mykernel")))
        

if __name__ == '__main__':
    unittest.main() 