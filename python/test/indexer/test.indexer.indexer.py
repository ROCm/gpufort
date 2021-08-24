#!/usr/bin/env python3
import time
import unittest
import cProfile,pstats,io

import addtoplevelpath
import indexer.indexer as indexer
import indexer.scoper as scoper
import utils.logging

log_format = "[%(levelname)s]\tgpufort:%(message)s"
log_level             = "debug2"
utils.logging.VERBOSE = False
utils.logging.init_logging("log.log",log_format,log_level)

gfortran_options="-DCUDA"

PROFILING_ENABLE = False

index = []

class TestIndexer(unittest.TestCase):
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
        if PROFILING_ENABLE:
            profiler = cProfile.Profile()
            profiler.enable()
        indexer.scan_file("test_modules.f90",gfortran_options,self._index)
        indexer.scan_file("test1.f90",gfortran_options,self._index)
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
        self._index.clear()
        indexer.load_gpufort_module_files(["./"],self._index)
    def test_4_indexer_check_modules_programs(self):
        self.assertEqual(len([mod for mod in self._index if mod["name"] == "simple"]),1, "Did not find module 'simple'")
        self.assertEqual(len([mod for mod in self._index if mod["name"] == "nested_subprograms"]),1, "Did not find module 'nested_subprograms'")
        self.assertEqual(len([mod for mod in self._index if mod["name"] == "simple"]),1, "Did not find module 'simple'")
    def test_5_indexer_check_program_test1(self):
        test1 = next((mod for mod in self._index if mod["name"] == "test1"),None)
        float_scalar = next((var for var in test1["variables"] if var["name"] == "float_scalar".lower()),None)
        self.assertIsNotNone(float_scalar)
        self.assertEqual(float_scalar["c_type"],"float")
        self.assertEqual(float_scalar["rank"],0)
        double_scalar = next((var for var in test1["variables"] if var["name"] == "double_scalar".lower()),None)
        self.assertIsNotNone(double_scalar)
        self.assertEqual(double_scalar["c_type"],"double")
        self.assertEqual(double_scalar["rank"],0)
        int_array2d = next((var for var in test1["variables"] if var["name"] == "int_array2d".lower()),None)
        self.assertIsNotNone(int_array2d)
        self.assertEqual(int_array2d["c_type"],"int")
        self.assertEqual(int_array2d["rank"],2)
        t = next((var for var in test1["variables"] if var["name"] == "t".lower()),None)
        self.assertIsNotNone(t)
    def test_6_indexer_check_module_simple(self):
        simple = next((mod for mod in self._index if mod["name"] == "simple"),None)
        a = next((var for var in simple["variables"] if var["name"] == "a".lower()),None)
        self.assertIsNotNone(a)
        self.assertEqual(a["c_type"],"int")
        self.assertEqual(a["rank"],0)
        n = next((var for var in simple["variables"] if var["name"] == "n".lower()),None)
        self.assertIsNotNone(n)
        self.assertEqual(n["c_type"],"int")
        self.assertEqual(n["rank"],0)
        self.assertTrue("parameter" in n["qualifiers"])
        self.assertEqual(n["value"].replace("(","").replace(")",""),"100")
        c = next((var for var in simple["variables"] if var["name"] == "c".lower()),None)
        self.assertIsNotNone(c)
        self.assertEqual(c["c_type"],"float")
        self.assertEqual(c["rank"],2)
        self.assertTrue("device" in c["qualifiers"])
        self.assertEqual(c["declare_on_target"],"alloc")
        mytype = next((typ for typ in simple["types"] if typ["name"] == "mytype".lower()),None)
        self.assertIsNotNone(mytype)
        b = next((var for var in mytype["variables"] if var["name"] == "b".lower()),None) # mytype#b
        self.assertIsNotNone(b)
        self.assertEqual(b["c_type"],"float")
        self.assertEqual(b["rank"],1)
        self.assertEqual(b["counts"][0],"n")
        self.assertEqual(b["lbounds"][0],"1")
    def test_7_indexer_check_module_nested_subprograms(self):
        nested_subprograms = next((mod for mod in self._index if mod["name"] == "nested_subprograms"),None)
        a = next((var for var in nested_subprograms["variables"] if var["name"] == "a".lower()),None)
        self.assertIsNotNone(a)
        self.assertEqual(a["c_type"],"int")
        self.assertEqual(a["rank"],0)
        n = next((var for var in nested_subprograms["variables"] if var["name"] == "n".lower()),None)
        self.assertIsNotNone(n)
        self.assertEqual(n["c_type"],"int")
        self.assertEqual(n["rank"],0)
        self.assertTrue("parameter" in n["qualifiers"])
        self.assertEqual(n["value"].replace("(","").replace(")",""),"1000")
        e = next((var for var in nested_subprograms["variables"] if var["name"] == "e".lower()),None)
        self.assertIsNotNone(e)
        self.assertEqual(e["c_type"],"float")
        self.assertEqual(e["rank"],2)
        self.assertEqual(e["counts"][0].replace(" ",""),"n-(-n)+1")
        self.assertEqual(e["counts"][1].replace(" ",""),"n-(-n)+1")
        self.assertEqual(e["lbounds"][0],"-n")
        self.assertEqual(e["lbounds"][1],"-n")
        self.assertFalse("device" in e["qualifiers"])
        self.assertFalse(e["declare_on_target"])
        mytype = next((typ for typ in nested_subprograms["types"] if typ["name"] == "mytype".lower()),None)
        self.assertIsNotNone(mytype)
        b = next((var for var in mytype["variables"] if var["name"] == "b".lower()),None) # mytype#b
        self.assertIsNotNone(b)
        self.assertEqual(b["c_type"],"double")
        self.assertEqual(b["rank"],1)
        self.assertEqual(b["counts"][0],"n")
        self.assertEqual(b["lbounds"][0],"1")
        # subprograms
        self.assertEqual(len(nested_subprograms["subprograms"]),2)
        func = next((sub for sub in nested_subprograms["subprograms"] if sub["name"] == "func".lower()),None)
        self.assertIsNotNone(func)
        self.assertEqual(func["kind"],"subroutine")
        self.assertEqual(len(func["subprograms"]),0)
        func2 = next((sub for sub in nested_subprograms["subprograms"] if sub["name"] == "func2".lower()),None)
        self.assertIsNotNone(func2)
        self.assertEqual(func2["kind"],"function")
        self.assertEqual(func2["result_name"],"res")
        self.assertEqual(len(func2["subprograms"]),2)
        # nested subprogram 1 
        func3 = next((sub for sub in func2["subprograms"] if sub["name"] == "func3".lower()),None)
        self.assertEqual(func3["kind"],"function")
        self.assertEqual(func3["result_name"],"func3")
        self.assertEqual(len(func3["subprograms"]),0)
        # nested subprogram 2
        func4 = next((sub for sub in func2["subprograms"] if sub["name"] == "func4".lower()),None)
        self.assertEqual(func4["kind"],"function")
        self.assertEqual(func4["result_name"],"func4")
        self.assertEqual(len(func4["subprograms"]),0)
        self.assertEqual(func4["attributes"],["host","device"])
      
if __name__ == '__main__':
    unittest.main() 