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
TEMPLATE = "test/fort2hip/templates/derived_types"
template_hip_cpp               = ENV.get_template(TEMPLATE+".template.hip.cpp")
template_f03                   = ENV.get_template(TEMPLATE+".template.f03")
template_copy_scalars_f03      = ENV.get_template(TEMPLATE+"_copy_scalars.template.f03")
template_copy_array_member_f03 = ENV.get_template(TEMPLATE+"_copy_array_member.template.f03")
template_size_bytes_f03        = ENV.get_template(TEMPLATE+"_size_bytes.template.f03")

testdata= \
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
"""

testdata_result_cpp = \
"""
typedef struct {
  double scalar_double;
  gpufort::array2<int> array_integer;
} basic;

typedef struct {
  basic single_basic;
  gpufort::array2<basic> array_basic;
  gpufort::array3<int> array_integer;
} nested;
"""

testdata_result_f03 = \
"""
type,bind(c) :: basic_interop
  real(8)              :: scalar_double
  type(gpufort_array2) :: array_integer
end type basic_interop

type,bind(c) :: nested_interop
  type(basic_interop)  :: single_basic
  type(gpufort_array2) :: array_basic
  type(gpufort_array3) :: array_integer
end type nested_interop
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
        SCOPE = indexer.indexerutils.create_scope_from_declaration_list(testdata)
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
    def test_1_render_cpp_structs(self):
        self.assertEqual(self.clean(template_hip_cpp.render(SCOPE)),\
            self.clean(testdata_result_cpp))
    def test_2_render_fortran_derived_types(self):
        #print(self.clean(template_f03.render(SCOPE)))
        #print(self.clean(testdata_result_f03))
        self.assertEqual(self.clean(template_f03.render(SCOPE)),\
            self.clean(testdata_result_f03))
    def test_3_render_fortran_size_bytes_routines(self):
        print(template_size_bytes_f03.render(SCOPE))
    def test_4_render_fortran_copy_scalars_routines(self):
        print(template_copy_scalars_f03.render(SCOPE))
    def test_5_render_fortran_copy_array_member_routines(self):
        print(template_copy_array_member_f03.render(SCOPE))

if __name__ == '__main__':
    unittest.main() 
