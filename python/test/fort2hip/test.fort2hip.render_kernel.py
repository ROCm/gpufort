#!/usr/bin/env python3
import os
import json
import jinja2
import unittest
import cProfile,pstats,io,time
import addtoplevelpath
import indexer.indexerutils
import utils.logging

LOG_FORMAT = "[%(levelname)s]\tgpufort:%(message)s"
utils.logging.VERBOSE    = False
utils.logging.init_logging("log.log",LOG_FORMAT,"warning")

PROFILING_ENABLE = False

ROOT   = os.path.realpath(os.path.join(os.path.dirname(__file__),"..",".."))
LOADER = jinja2.FileSystemLoader(ROOT)
ENV    = jinja2.Environment(loader=LOADER, trim_blocks=True,
           lstrip_blocks=True, undefined=jinja2.StrictUndefined)

TEST_TEMPLATE_DIR = os.path.join("test","fort2hip","templates")
render_kernel_template              = ENV.get_template(os.path.join(TEST_TEMPLATE_DIR,"render_gpu_kernel.template.hip.cpp"))
render_gpu_kernel_launcher_template = ENV.get_template(os.path.join(TEST_TEMPLATE_DIR,"render_gpu_kernel_launcher.template.hip.cpp"))
render_cpu_kernel_launcher_template = ENV.get_template(os.path.join(TEST_TEMPLATE_DIR,"render_cpu_kernel_launcher.template.hip.cpp"))

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
"""

SCOPE = None

class TestRenderKernel(unittest.TestCase):
    def prepare(self,text):
        return text.strip().split("\n")
    def clean(self,text):
        return text.replace(" ","").replace("\t","").replace("\n","").replace("\r","")
    def setUp(self):
        global SCOPE
        global PROFILING_ENABLE
        if PROFILING_ENABLE:
            self._profiler = cProfile.Profile()
            self._profiler.enable()
        self._started_at = time.time()
        SCOPE = indexer.indexerutils.create_scope_from_declaration_list(declaration_list)
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
    def create_kernel_context(self,
        kernel_name,
        scope):
        local_vars          = [ivar for ivar in scope["variables"] if "_local" in ivar["name"]]
        shared_vars         = [ivar for ivar in scope["variables"] if "_shared" in ivar["name"]]
        global_vars         = [ivar for ivar in scope["variables"] if ivar not in local_vars and\
                              ivar not in shared_vars]
        global_reduced_vars = [] 
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
    def create_gpu_kernel_launcher_context(self,
            kernel_name,kind="auto",
            generate_debug_code=True):
        # for launcher interface
        kernel_launcher["kind"]                = "auto"
        kernel_launcher["name"]                = "_".join("launch",kernel_name,kind) 
        kernel_launcher["block"]               = [1024,1,1]
        kernel_launcher["grid"]                = [4,1,1] 
        kernel_launcher["problem_size"]        = [4096,1,1] 
        kernel_launcher["generate_debug_code"] = True
    def create_cpu_kernel_launcher_context(self,kernel_name,kind="auto"):
        # for launcher interface
        kernel_launcher["launcher_name"]       = "_".join(["launch",kernel["name"]])
        kernel_launcher["block"]               = [1024,1,1]
        kernel_launcher["grid"]                = [4,1,1] 
        kernel_launcher["problem_size"]        = [4096,1,1] 
        kernel_launcher["generate_debug_code"] = True
    def test_1_render_kernel(self):
        context = {"kernel":self.create_kernel_context()}
        print(render_kernel_template.render(context))
    def test_2_render_gpu_kernel_launcher(self):
        context = {"kernel":self.create_kernel_context(),"launcher_kind":"auto"}
        print(render_gpu_kernel_launcher_template.render(context))
    def test_3_render_cpu_kernel_launcher(self):
        context = {
         "kernel":self.create_kernel_context(),
         "kernel_launcher": self.create_gpu_kernel_launcher_context( 
        }
        print(render_gpu_kernel_launcher_template.render(context))
        

if __name__ == '__main__':
    unittest.main() 
