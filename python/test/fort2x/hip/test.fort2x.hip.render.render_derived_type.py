#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import os
import json
import jinja2
import unittest
import cProfile,pstats,io,time
import addtoplevelpath
import indexer.indexerutils
import utils.logging

import fort2x.hip.render

LOG_FORMAT = "[%(levelname)s]\tgpufort:%(message)s"
utils.logging.VERBOSE    = False
utils.logging.init_logging("log.log",LOG_FORMAT,"warning")

PROFILING_ENABLE = False

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

class TestRenderDerivedType(unittest.TestCase):
    def prepare(self,text):
        return text.strip().split("\n")
    def clean(self,text):
        return text.replace(" ","").replace("\t","").replace("\n","").replace("\r","")
    def setUp(self):
        global PROFILING_ENABLE
        if PROFILING_ENABLE:
            self._profiler = cProfile.Profile()
            self._profiler.enable()
        self._started_at = time.time()
        self._scope      = indexer.indexerutils.create_scope_from_declaration_list(testdata)
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
    def test_1_render_derived_types_cpp(self):
        #print(fort2x.hip.render.render_derived_types_cpp(self._scope["types"]))
        self.assertEqual(self.clean(fort2x.hip.render.render_derived_types_cpp(self._scope["types"])),\
            self.clean(testdata_result_cpp))
    def test_2_render_derived_types_f03(self):
        print(fort2x.hip.render.render_derived_types_f03(self._scope["types"]))
        print(testdata_result_f03)
        self.assertEqual(self.clean(fort2x.hip.render.render_derived_types_f03(self._scope["types"])),\
            self.clean(testdata_result_f03))
    def test_3_render_derived_type_size_bytes_routines_f03(self):
        print(fort2x.hip.render.render_derived_type_size_bytes_routines_f03(self._scope["types"]))
    def test_4_render_derived_type_copy_scalars_routines_f03(self):
        print(fort2x.hip.render.render_derived_type_copy_scalars_routines_f03(self._scope["types"]))
    def test_5_render_derived_type_copy_array_member_routines_f03(self):
        print(fort2x.hip.render.render_derived_type_copy_array_member_routines_f03(self._scope["types"]))

if __name__ == '__main__':
    unittest.main() 
