#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import time
import unittest
import cProfile,pstats,io

import addtoplevelpath
from gpufort import indexer
from gpufort import util
from gpufort import linemapper

log_format = "[%(levelname)s]\tgpufort:%(message)s"
log_level                   = "debug"
util.logging.opts.verbose   = False
#util.logging.opts.log_filter = "acc"
util.logging.opts.traceback = False
util.logging.init_logging("log.log",log_format,log_level)

preproc_options="-DCUDA"

PROFILING_ENABLE          = False

index = []

class TestIndexer(unittest.TestCase):
    def setUp(self):
        global index
        self.index = index
        self.started_at = time.time()
    def tearDown(self):
        elapsed = time.time() - self.started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 6)))
    def test_0_donothing(self):
        pass 
    def test_1_indexer_scan_files(self):
        global PROFILING_ENABLE
        global USE_EXTERNAL_PREPROCESSOR
        
        if PROFILING_ENABLE:
            profiler = cProfile.Profile()
            profiler.enable()
        indexer.update_index_from_linemaps(linemapper.read_file("test_modules_1.f90",preproc_options=preproc_options),self.index)
        indexer.update_index_from_linemaps(linemapper.read_file("test_modules_2.f90",preproc_options=preproc_options),self.index)
        indexer.update_index_from_linemaps(linemapper.read_file("test1.f90",preproc_options=preproc_options),self.index)
        if PROFILING_ENABLE:
            profiler.disable() 
            s = io.StringIO()
            sortby = 'cumulative'
            stats = pstats.Stats(profiler, stream=s).sort_stats(sortby)
            stats.print_stats(10)
            print(s.getvalue())
    def test_2_indexer_write_module_files(self):
        indexer.write_gpufort_module_files(index,"./")
    def test_3_indexer_load_module_files(self):
        self.index.clear()
        indexer.load_gpufort_module_files(["./"],self.index)
    def test_4_indexer_check_modules_programs(self):
        self.assertEqual(len([mod for mod in self.index if mod["name"] == "simple"]),1, "Did not find module 'simple'")
        self.assertEqual(len([mod for mod in self.index if mod["name"] == "nested_procedures"]),1, "Did not find module 'nested_procedures'")
        self.assertEqual(len([mod for mod in self.index if mod["name"] == "simple"]),1, "Did not find module 'simple'")
    def test_5_indexer_check_program_test1(self):
        test1 = next((mod for mod in self.index if mod["name"] == "test1"),None)
        float_scalar = next((var for var in test1["variables"] if var["name"] == "float_scalar".lower()),None)
        self.assertIsNotNone(float_scalar)
#        self.assertEqual(float_scalar["c_type"],"float")
        self.assertEqual(float_scalar["rank"],0)
        double_scalar = next((var for var in test1["variables"] if var["name"] == "double_scalar".lower()),None)
        self.assertIsNotNone(double_scalar)
#        self.assertEqual(double_scalar["c_type"],"double")
        self.assertEqual(double_scalar["rank"],0)
        int_array2d = next((var for var in test1["variables"] if var["name"] == "int_array2d".lower()),None)
        self.assertIsNotNone(int_array2d)
#        self.assertEqual(int_array2d["c_type"],"int")
        self.assertEqual(int_array2d["rank"],2)
        t = next((var for var in test1["variables"] if var["name"] == "t".lower()),None)
        self.assertIsNotNone(t)
    def test_6_indexer_check_module_simple(self):
        simple = next((mod for mod in self.index if mod["name"] == "simple"),None)
        a = next((var for var in simple["variables"] if var["name"] == "a".lower()),None)
        self.assertIsNotNone(a)
#        self.assertEqual(a["c_type"],"int")
        self.assertEqual(a["rank"],0)
        n = next((var for var in simple["variables"] if var["name"] == "n".lower()),None)
        self.assertIsNotNone(n)
#        self.assertEqual(n["c_type"],"int")
        self.assertEqual(n["rank"],0)
        self.assertTrue("parameter" in n["attributes"])
        #self.assertEqual(n["value"].replace("(","").replace(")",""),"100")
        c = next((var for var in simple["variables"] if var["name"] == "c".lower()),None)
        self.assertIsNotNone(c)
#        self.assertEqual(c["c_type"],"float")
        self.assertEqual(c["rank"],2)
        self.assertTrue("device" in c["attributes"])
        self.assertEqual(c["declare_on_target"],"alloc")
        mytype = next((typ for typ in simple["types"] if typ["name"] == "mytype".lower()),None)
        self.assertIsNotNone(mytype)
        b = next((var for var in mytype["variables"] if var["name"] == "b".lower()),None) # mytype#b
        self.assertIsNotNone(b)
#        self.assertEqual(b["c_type"],"float")
        self.assertEqual(b["rank"],1)
        #self.assertEqual(b["counts"][0],"n")
        #self.assertEqual(b["lbounds"][0],"1")
    def test_7_indexer_check_module_nested_procedures(self):
        nested_procedures = next((mod for mod in self.index if mod["name"] == "nested_procedures"),None)
        a = next((var for var in nested_procedures["variables"] if var["name"] == "a".lower()),None)
        self.assertIsNotNone(a)
#        self.assertEqual(a["c_type"],"int")
        self.assertEqual(a["rank"],0)
        n = next((var for var in nested_procedures["variables"] if var["name"] == "n".lower()),None)
        self.assertIsNotNone(n)
#        self.assertEqual(n["c_type"],"int")
        self.assertEqual(n["rank"],0)
        self.assertTrue("parameter" in n["attributes"])
        #self.assertEqual(n["value"].replace("(","").replace(")",""),"1000")
        e = next((var for var in nested_procedures["variables"] if var["name"] == "e".lower()),None)
        self.assertIsNotNone(e)
#        self.assertEqual(e["c_type"],"float")
        self.assertEqual(e["rank"],2)
        #self.assertEqual(e["counts"][0].replace(" ",""),"n-(-n)+1")
        #self.assertEqual(e["counts"][1].replace(" ",""),"n-(-n)+1")
        #self.assertEqual(e["lbounds"][0],"-n")
        #self.assertEqual(e["lbounds"][1],"-n")
        self.assertFalse("device" in e["attributes"])
        self.assertFalse(e["declare_on_target"])
        mytype = next((typ for typ in nested_procedures["types"] if typ["name"] == "mytype".lower()),None)
        self.assertIsNotNone(mytype)
        b = next((var for var in mytype["variables"] if var["name"] == "b".lower()),None) # mytype#b
        self.assertIsNotNone(b)
#        self.assertEqual(b["c_type"],"double")
        self.assertEqual(b["rank"],1)
        #self.assertEqual(b["counts"][0],"n")
        #self.assertEqual(b["lbounds"][0],"1")
        # procedures
        self.assertEqual(len(nested_procedures["procedures"]),2)
        func = next((sub for sub in nested_procedures["procedures"] if sub["name"] == "func".lower()),None)
        self.assertIsNotNone(func)
        self.assertEqual(func["kind"],"subroutine")
        self.assertEqual(len(func["procedures"]),0)
        func2 = next((sub for sub in nested_procedures["procedures"] if sub["name"] == "func2".lower()),None)
        self.assertIsNotNone(func2)
        self.assertEqual(func2["kind"],"function")
        self.assertEqual(func2["result_name"],"res")
        self.assertEqual(len(func2["procedures"]),2)
        # nested procedure 1
        func3 = next((sub for sub in func2["procedures"] if sub["name"] == "func3".lower()),None)
        self.assertEqual(func3["kind"],"function")
        self.assertEqual(func3["result_name"],"func3")
        self.assertEqual(len(func3["procedures"]),0)
        # nested procedure 2
        func4 = next((sub for sub in func2["procedures"] if sub["name"] == "func4".lower()),None)
        self.assertEqual(func4["kind"],"function")
        self.assertEqual(func4["result_name"],"func4")
        self.assertEqual(len(func4["procedures"]),0)
        self.assertEqual(func4["attributes"],["host","device"])
    def test_8_indexer_check_accessibility(self):
        private_mod1 = next((mod for mod in self.index if mod["name"] == "private_mod1"),None)
        self.assertIsNotNone(private_mod1)
        self.assertTrue(private_mod1["accessibility"] == "private")
        # subroutines
        private_proc1 = next((sub for sub in private_mod1["procedures"] if sub["name"] == "private_proc1".lower()),None)
        public_proc1 = next((sub for sub in private_mod1["procedures"] if sub["name"] == "public_proc1".lower()),None)
        self.assertIsNotNone(private_proc1)
        self.assertIsNotNone(public_proc1)
        self.assertTrue(indexer.scope._get_accessibility(private_proc1,private_mod1) == "private")
        self.assertFalse(indexer.scope._get_accessibility(private_proc1,private_mod1) == "public")
        self.assertTrue(indexer.scope._get_accessibility(public_proc1,private_mod1) == "public")
        self.assertFalse(indexer.scope._get_accessibility(public_proc1,private_mod1) == "private")
        # types
        public_type1 = next((sub for sub in private_mod1["types"] if sub["name"] == "public_type1".lower()),None)
        public_type2 = next((sub for sub in private_mod1["types"] if sub["name"] == "public_type2".lower()),None)
        private_type1 = next((sub for sub in private_mod1["types"] if sub["name"] == "private_type1".lower()),None)
        self.assertIsNotNone(public_type1)
        self.assertIsNotNone(public_type2)
        self.assertIsNotNone(private_type1)
        self.assertTrue(indexer.scope._get_accessibility(public_type1,private_mod1) == "public")
        self.assertTrue(indexer.scope._get_accessibility(public_type2,private_mod1) == "public")
        self.assertTrue(indexer.scope._get_accessibility(private_type1,private_mod1) == "private")
        self.assertFalse(indexer.scope._get_accessibility(public_type1,private_mod1) == "private")
        self.assertFalse(indexer.scope._get_accessibility(public_type2,private_mod1) == "private")
        self.assertFalse(indexer.scope._get_accessibility(private_type1,private_mod1) == "public")
        # variables
        public_var1 = next((sub for sub in private_mod1["variables"] if sub["name"] == "public_var1".lower()),None)
        public_var2 = next((sub for sub in private_mod1["variables"] if sub["name"] == "public_var2".lower()),None)
        private_var1 = next((sub for sub in private_mod1["variables"] if sub["name"] == "private_var1".lower()),None)
        self.assertIsNotNone(public_var1)
        self.assertIsNotNone(public_var2)
        self.assertIsNotNone(private_var1)
        self.assertTrue(indexer.scope._get_accessibility(public_var1,private_mod1) == "public")
        self.assertTrue(indexer.scope._get_accessibility(public_var2,private_mod1) == "public")
        self.assertTrue(indexer.scope._get_accessibility(private_var1,private_mod1) == "private")
        self.assertFalse(indexer.scope._get_accessibility(public_var1,private_mod1) == "private")
        self.assertFalse(indexer.scope._get_accessibility(public_var2,private_mod1) == "private")
        self.assertFalse(indexer.scope._get_accessibility(private_var1,private_mod1) == "public")
        #
        for modname in ["public_mod1","public_mod2"]:
            public_mod = next((mod for mod in self.index if mod["name"] == modname),None)
            self.assertIsNotNone(public_mod)
            self.assertTrue(public_mod["accessibility"] == "public")
            # subroutines
            private_proc1 = next((sub for sub in public_mod["procedures"] if sub["name"] == "private_proc1".lower()),None)
            public_proc1 = next((sub for sub in public_mod["procedures"] if sub["name"] == "public_proc1".lower()),None)
            self.assertIsNotNone(private_proc1)
            self.assertIsNotNone(public_proc1)
            self.assertTrue(indexer.scope._get_accessibility(private_proc1,public_mod) == "private")
            self.assertTrue(indexer.scope._get_accessibility(public_proc1,public_mod) == "public")
            self.assertFalse(indexer.scope._get_accessibility(private_proc1,public_mod) == "public")
            self.assertFalse(indexer.scope._get_accessibility(public_proc1,public_mod) == "private")
            # types
            private_type1 = next((sub for sub in public_mod["types"] if sub["name"] == "private_type1".lower()),None)
            private_type2 = next((sub for sub in public_mod["types"] if sub["name"] == "private_type2".lower()),None)
            public_type1 = next((sub for sub in public_mod["types"] if sub["name"] == "public_type1".lower()),None)
            self.assertIsNotNone(private_type1)
            self.assertIsNotNone(private_type2)
            self.assertIsNotNone(public_type1)
            self.assertTrue(indexer.scope._get_accessibility(private_type1,public_mod) == "private")
            self.assertTrue(indexer.scope._get_accessibility(private_type2,public_mod) == "private")
            self.assertTrue(indexer.scope._get_accessibility(public_type1,public_mod) == "public")
            self.assertFalse(indexer.scope._get_accessibility(private_type1,public_mod) == "public")
            self.assertFalse(indexer.scope._get_accessibility(private_type2,public_mod) == "public")
            self.assertFalse(indexer.scope._get_accessibility(public_type1,public_mod) == "private")
            # variables
            private_var1 = next((sub for sub in public_mod["variables"] if sub["name"] == "private_var1".lower()),None)
            private_var2 = next((sub for sub in public_mod["variables"] if sub["name"] == "private_var2".lower()),None)
            public_var1 = next((sub for sub in public_mod["variables"] if sub["name"] == "public_var1".lower()),None)
            self.assertIsNotNone(private_var1)
            self.assertIsNotNone(private_var2)
            self.assertIsNotNone(public_var1)
            self.assertTrue(indexer.scope._get_accessibility(private_var1,public_mod) == "private")
            self.assertTrue(indexer.scope._get_accessibility(private_var2,public_mod) == "private")
            self.assertTrue(indexer.scope._get_accessibility(public_var1,public_mod) == "public")
            self.assertFalse(indexer.scope._get_accessibility(private_var1,public_mod) == "public")
            self.assertFalse(indexer.scope._get_accessibility(private_var2,public_mod) == "public")
            self.assertFalse(indexer.scope._get_accessibility(public_var1,public_mod) == "private")
      
if __name__ == '__main__':
    unittest.main() 
