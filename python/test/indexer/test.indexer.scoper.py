#!/usr/bin/env python3
import time
import unittest

import addtoplevelpath
import indexer.indexer as indexer
import translator.translator as translator
import indexer.scoper as scoper
import utils.logging

utils.logging.VERBOSE    = True
utils.logging.initLogging("log.log","debug2")

gfortranOptions="-DCUDA"

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
        indexer.scanFile("test_modules.f90",gfortranOptions,self._index)
        indexer.scanFile("test1.f90",gfortranOptions,self._index)
    def test_2_scoper_search_for_variables(self):
        c   = scoper.searchIndexForVariable(index,"test1","c") # included from module 'simple'
        scoper.SCOPES.clear()
        t_b = scoper.searchIndexForVariable(index,"test1",\
          translator.createIndexSearchTagForVariable("t%b")) # type of t included from module 'simple'
        scoper.SCOPES.clear()
        tc_t1list_a = scoper.searchIndexForVariable(index,"test1",\
          translator.createIndexSearchTagForVariable("tc%t1list(i)%a")) # type of t included from module 'simple'
        scoper.SCOPES.clear()
        tc_t2list_t1list_a = scoper.searchIndexForVariable(index,"test1",
          translator.createIndexSearchTagForVariable("tc%t2list(indexlist%j)%t1list(i)%a")) 
        scoper.SCOPES.clear()
    def test_3_scoper_search_for_variables_reuse_scope(self):
        c   = scoper.searchIndexForVariable(index,"test1","c") # included from module 'simple'
        t_b = scoper.searchIndexForVariable(index,"test1",\
          translator.createIndexSearchTagForVariable("t%b")) # type of t included from module 'simple'
        tc_t1list_a = scoper.searchIndexForVariable(index,"test1",\
          translator.createIndexSearchTagForVariable("tc%t1list(i)%a")) # type of t included from module 'simple'
        tc_t2list_t1list_a = scoper.searchIndexForVariable(index,"test1",
          translator.createIndexSearchTagForVariable("tc%t2list(indexlist%j)%t1list(i)%a")) 
        scoper.SCOPES.clear()
    def test_4_scoper_search_for_subprograms(self):
        func2 = scoper.searchIndexForSubprogram(index,"test1","func2")
        scoper.SCOPES.clear()
        func3 = scoper.searchIndexForSubprogram(index,"nested_subprograms:func2","func3")
        scoper.SCOPES.clear()
        type1 = scoper.searchIndexForType(index,"complex_types","type1")
        scoper.SCOPES.clear()

if __name__ == '__main__':
    unittest.main() 
