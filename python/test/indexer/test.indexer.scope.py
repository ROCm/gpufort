#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import time
import unittest

import addtoplevelpath
from gpufort import indexer
from gpufort import translator
from gpufort import linemapper
from gpufort import util

util.logging.opts.verbose = False
LOG_FORMAT = "[%(levelname)s]\tgpufort:%(message)s"
util.logging.init_logging("log.log",LOG_FORMAT,"warning")

gfortran_options="-DCUDA"

indexer.opts.error_handling="strict"

# scan index
index = []

# main file
class TestScoper(unittest.TestCase):
    def setUp(self):
        global index
        self.started_at = time.time()
    def tearDown(self):
        elapsed = time.time() - self.started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 6)))
    def test_0_donothing(self):
        pass 
    def test_1_indexer_scan_files(self):
        indexer.update_index_from_linemaps(linemapper.read_file("test_modules.f90",gfortran_options),index)
        indexer.update_index_from_linemaps(linemapper.read_file("test1.f90",gfortran_options),index)
    def test_2_scope_search_for_vars(self):
        c   = indexer.scope.search_index_for_var(index,"test1","c") # included from module 'simple'
        indexer.opts.scopes.clear()
        t_b = indexer.scope.search_index_for_var(index,"test1",\
          "t%b") # type of t included from module 'simple'
        indexer.opts.scopes.clear()
        tc_t1list_a = indexer.scope.search_index_for_var(index,"test1","tc%t1list(i)%a") # type of t included from module 'simple'
        indexer.opts.scopes.clear()
        tc_t2list_t1list_a = indexer.scope.search_index_for_var(index,"test1","tc%t2list(indexlist%j)%t1list(i)%a") 
        indexer.opts.scopes.clear()
    def test_3_scope_search_for_vars_reuse_scope(self):
        c   = indexer.scope.search_index_for_var(index,"test1","c") # included from module 'simple'
        t_b = indexer.scope.search_index_for_var(index,"test1","t%b") # type of t included from module 'simple'
        tc_t1list_a = indexer.scope.search_index_for_var(index,"test1","tc%t1list(i)%a") # type of t included from module 'simple'
        tc_t2list_t1list_a = indexer.scope.search_index_for_var(index,"test1",\
          "tc%t2list(indexlist%j)%t1list(i)%a") 
        indexer.opts.scopes.clear()
    def test_4_scope_search_for_procedures(self):
        func2 = indexer.scope.search_index_for_procedure(index,"test1","func2")
        func3 = indexer.scope.search_index_for_procedure(index,"nested_procedures:func2","func3")
        type1 = indexer.scope.search_index_for_type(index,"complex_types","type1")
        indexer.opts.scopes.clear()
    def test_5_scope_search_for_top_level_procedures(self):
        func2 = indexer.scope.search_index_for_procedure(index,"test1","top_level_subroutine")
        indexer.opts.scopes.clear()

if __name__ == '__main__':
    unittest.main() 
