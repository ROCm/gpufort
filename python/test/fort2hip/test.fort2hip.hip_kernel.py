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

# TEMPLATE
TEMPLATE         = "test/fort2hip/templates/hip_kernel"
template_hip_cpp = ENV.get_template(TEMPLATE+".template.hip.cpp")

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

class TestDerivedTypes(unittest.TestCase):
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
    def test_0(self):
        local_vars = [ivar for ivar in SCOPE["variables"] if "_local" in ivar["name"]]
        shared_vars = [ivar for ivar in SCOPE["variables"] if "_shared" in ivar["name"]]
        params     = [ivar for ivar in SCOPE["variables"] if ivar not in local_vars and\
                      ivar not in shared_vars]
        print([ivar["name"] for ivar in local_vars])
        print([ivar["name"] for ivar in shared_vars])
        print([ivar["name"] for ivar in params])
        kernel = {}
        kernel["launch_bounds"] = "__launch_bounds___(1024,1)"
        kernel["name"]          = "mykernel"
        kernel["params"]        = params
        kernel["local_vars"]    = local_vars
        kernel["shared_vars"]   = shared_vars
        kernel["c_body"]        = "// do nothing" 
        kernel["f_body"]        = "! do nothing"
        print(template_hip_cpp.render({"kernel":kernel}))
    #    self.assertEqual(self.clean(template_hip_cpp.render(SCOPE)),\
    #        self.clean(testdata_result_cpp))
    #def test_2_render_fortran_derived_types(self):
    #    #print(self.clean(template_f03.render(SCOPE)))
    #    #print(self.clean(testdata_result_f03))
    #    self.assertEqual(self.clean(template_f03.render(SCOPE)),\
    #        self.clean(testdata_result_f03))
    #def test_3_render_fortran_size_bytes_routines(self):
    #    print(template_size_bytes_f03.render(SCOPE))
    #def test_4_render_fortran_copy_scalars_routines(self):
    #    print(template_copy_scalars_f03.render(SCOPE))
    #def test_5_render_fortran_copy_array_member_routines(self):
    #    print(template_copy_array_member_f03.render(SCOPE))

if __name__ == '__main__':
    unittest.main() 
