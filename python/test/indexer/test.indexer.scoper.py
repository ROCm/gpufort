#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import time
import unittest

import addtoplevelpath
import indexer.indexer as indexer
import translator.translator as translator
import indexer.scoper as scoper
import linemapper.linemapper as linemapper
import utils.logging

utils.logging.VERBOSE = False
LOG_FORMAT = "[%(levelname)s]\tgpufort:%(message)s"
utils.logging.init_logging("log.log",LOG_FORMAT,"warning")

gfortran_options="-DCUDA"

scoper.ERROR_HANDLING="strict"

USE_EXTERNAL_PREPROCESSOR = False

# scan index
index = []

# main file
class TestScoper(unittest.TestCase):
    def setUp(self):
        global index
        self._index = index
        self._started_at = time.time()
    def tearDown(self):
        elapsed = time.time() - self._started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 6)))
    def test_0_donothing(self):
        pass 
    def test_1_indexer_scan_files(self):
        global USE_EXTERNAL_PREPROCESSOR
        if USE_EXTERNAL_PREPROCESSOR:
            indexer.scan_file("test_modules.f90",gfortran_options,self._index)
            indexer.scan_file("test1.f90",gfortran_options,self._index)
        else:
            indexer.update_index_from_linemaps(linemapper.read_file("test_modules.f90",gfortran_options),self._index)
            indexer.update_index_from_linemaps(linemapper.read_file("test1.f90",gfortran_options),self._index)
    def test_2_scoper_search_for_variables(self):
        c   = scoper.search_index_for_variable(index,"test1","c") # included from module 'simple'
        scoper.SCOPES.clear()
        t_b = scoper.search_index_for_variable(index,"test1",\
          "t%b") # type of t included from module 'simple'
        scoper.SCOPES.clear()
        tc_t1list_a = scoper.search_index_for_variable(index,"test1","tc%t1list(i)%a") # type of t included from module 'simple'
        scoper.SCOPES.clear()
        tc_t2list_t1list_a = scoper.search_index_for_variable(index,"test1","tc%t2list(indexlist%j)%t1list(i)%a") 
        scoper.SCOPES.clear()
    def test_3_scoper_search_for_variables_reuse_scope(self):
        c   = scoper.search_index_for_variable(index,"test1","c") # included from module 'simple'
        t_b = scoper.search_index_for_variable(index,"test1","t%b") # type of t included from module 'simple'
        tc_t1list_a = scoper.search_index_for_variable(index,"test1","tc%t1list(i)%a") # type of t included from module 'simple'
        tc_t2list_t1list_a = scoper.search_index_for_variable(index,"test1",\
          "tc%t2list(indexlist%j)%t1list(i)%a") 
        scoper.SCOPES.clear()
    def test_4_scoper_search_for_subprograms(self):
        func2 = scoper.search_index_for_subprogram(index,"test1","func2")
        func3 = scoper.search_index_for_subprogram(index,"nested_subprograms:func2","func3")
        type1 = scoper.search_index_for_type(index,"complex_types","type1")
        scoper.SCOPES.clear()
    def test_5_scoper_search_for_top_level_subprograms(self):
        func2 = scoper.search_index_for_subprogram(index,"test1","top_level_subroutine")
        scoper.SCOPES.clear()

if __name__ == '__main__':
    unittest.main() 