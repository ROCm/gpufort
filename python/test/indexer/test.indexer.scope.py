#!/usr/bin/env python3

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

preproc_options="-DCUDA"

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
        indexer.update_index_from_linemaps(linemapper.read_file("test_modules_1.f90",preproc_options=preproc_options),index)
        indexer.update_index_from_linemaps(linemapper.read_file("test_modules_2.f90",preproc_options=preproc_options),index)
        indexer.update_index_from_linemaps(linemapper.read_file("test1.f90",preproc_options=preproc_options),index)
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
    def test_3_scope_search_for_implicitly_declared_vars(self):
        _i0   = indexer.scope.search_index_for_var(index,"test1","_i0") # included from module 'simple'
        indexer.opts.scopes.clear()
    def test_4_scope_search_for_vars_reuse_scope(self):
        c   = indexer.scope.search_index_for_var(index,"test1","c") # included from module 'simple'
        t_b = indexer.scope.search_index_for_var(index,"test1","t%b") # type of t included from module 'simple'
        tc_t1list_a = indexer.scope.search_index_for_var(index,"test1","tc%t1list(i)%a") # type of t included from module 'simple'
        tc_t2list_t1list_a = indexer.scope.search_index_for_var(index,"test1",\
          "tc%t2list(indexlist%j)%t1list(i)%a") 
        indexer.opts.scopes.clear()
    def test_5_scope_search_for_procedures(self):
        func2 = indexer.scope.search_index_for_procedure(index,"test1","func2")
        func3 = indexer.scope.search_index_for_procedure(index,"nested_procedures:func2","func3")
        func4 = indexer.scope.search_index_for_procedure(index,"nested_procedures:func2","func4")
        func4 = indexer.scope.search_index_for_procedure(index,"nested_procedures:func2:func3","func4")
        type1 = indexer.scope.search_index_for_type(index,"complex_types","type1")
        indexer.opts.scopes.clear()
    def test_6_scope_search_for_top_level_procedures(self):
        func2 = indexer.scope.search_index_for_procedure(index,"test1","top_level_subroutine")
        indexer.opts.scopes.clear()
    def test_7_scope_check_accessibility(self):
        public_proc1  = indexer.scope.search_index_for_procedure(index,"test1","public_proc1")
        try:
            private_proc1 = indexer.scope.search_index_for_procedure(index,"test1","private_proc1")
            self.assertTrue(False)
        except util.error.LookupError:
            pass
        # types
        public_type1 = indexer.scope.search_index_for_type(index,"test1","public_type1")
        public_type2 = indexer.scope.search_index_for_type(index,"test1","public_type1")
        try:
            private_type1 = indexer.scope.search_index_for_type(index,"test1","private_type1")
            self.assertTrue(False)
        except util.error.LookupError:
            pass
        public_var1  = indexer.scope.search_index_for_var(index,"test1","public_var1")
        public_var2  = indexer.scope.search_index_for_var(index,"test1","public_var1")
        try:
            private_var1 = indexer.scope.search_index_for_var(index,"test1","private_var1")
            self.assertTrue(False)
        except util.error.LookupError:
            pass
        indexer.opts.scopes.clear()
    def test_8_combine_only_use_statements(self):
        use_statements=[
          "use a, only: b1 => a1",
          "use a, only: b2 => a2",
          "use a, only: b3 => a3",
        ]
        result = [
          {'name': 'a',
          'only': [{'original': 'a1', 'renamed': 'b1'},
                   {'original': 'a2', 'renamed': 'b2'},
                   {'original': 'a3', 'renamed': 'b3'}],
          'attributes': [],
          'renamings': []}
        ]
        iused_modules = []
        for stmt in use_statements:
            iused_modules.append(indexer.create_index_record_from_use_statement(stmt))
        iused_modules = indexer.scope.combine_use_statements(iused_modules)
        self.assertEqual(len(iused_modules),len(result))
        for i in range(0,len(iused_modules)):
            self.assertEqual(iused_modules[i]["name"],result[i]["name"])
            self.assertEqual(iused_modules[i]["attributes"],result[i]["attributes"])
            self.assertEqual(iused_modules[i]["renamings"],result[i]["renamings"])
            self.assertEqual(iused_modules[i]["only"],result[i]["only"])
    def test_9_combone_mixed_use_statements(self):
        use_statements=[
          "use a, b1 => a1",
          "use a, b2 => a2",
          "use a",
          "use a, b3 => a3",
          "use a, only: b4 => a4",
        ]
        result = [
         {'name': 'a',
         'attributes': [], 
         'renamings': 
            [{'original': 'a1', 'renamed': 'b1'}, 
             {'original': 'a2', 'renamed': 'b2'}, 
             {'original': 'a3', 'renamed': 'b3'}, 
             {'original': 'a4', 'renamed': 'b4'}], 
         'only': []} 
        ]
        iused_modules = []
        for stmt in use_statements:
            iused_modules.append(indexer.create_index_record_from_use_statement(stmt))
        iused_modules = indexer.scope.combine_use_statements(iused_modules)
        self.assertEqual(len(iused_modules),len(result))
        for i in range(0,len(iused_modules)):
            self.assertEqual(iused_modules[i]["name"],result[i]["name"])
            self.assertEqual(iused_modules[i]["attributes"],result[i]["attributes"])
            self.assertEqual(iused_modules[i]["renamings"],result[i]["renamings"])
            self.assertEqual(iused_modules[i]["only"],result[i]["only"])

if __name__ == '__main__':
    unittest.main() 
