#!/usr/bin/env python3
import time
import unittest

import addtoplevelpath
import indexer.indexer as indexer
import indexer.scoper as scoper
import utils.logging

LOG_FORMAT = "[%(levelname)s]\tgpufort:%(message)s"
utils.logging.VERBOSE    = False
utils.logging.initLogging("log.log",LOG_FORMAT,"warning")
        
gfortranOptions="-DCUDA"

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
        indexer.scanFile("test_modules.f90",gfortranOptions,self._index)
        indexer.scanFile("test1.f90",gfortranOptions,self._index)
    def test_2_indexer_write_module_files(self):
        indexer.writeGpufortModuleFiles(index,"./")
    def test_3_indexer_load_module_files(self):
        self._index.clear()
        indexer.loadGpufortModuleFiles(["./"],self._index)
    def test_4_indexer_check_modules_programs(self):
        self.assertEqual(len([mod for mod in self._index if mod["name"] == "simple"]),1, "Did not find module 'simple'")
        self.assertEqual(len([mod for mod in self._index if mod["name"] == "nested_subprograms"]),1, "Did not find module 'nested_subprograms'")
        self.assertEqual(len([mod for mod in self._index if mod["name"] == "simple"]),1, "Did not find module 'simple'")
    def test_5_indexer_check_program_test1(self):
        test1 = next((mod for mod in self._index if mod["name"] == "test1"),None)
        floatScalar = next((var for var in test1["variables"] if var["name"] == "floatScalar".lower()),None)
        self.assertIsNotNone(floatScalar)
        self.assertEqual(floatScalar["cType"],"float")
        self.assertEqual(floatScalar["rank"],0)
        doubleScalar = next((var for var in test1["variables"] if var["name"] == "doubleScalar".lower()),None)
        self.assertIsNotNone(doubleScalar)
        self.assertEqual(doubleScalar["cType"],"double")
        self.assertEqual(doubleScalar["rank"],0)
        intArray2d = next((var for var in test1["variables"] if var["name"] == "intArray2d".lower()),None)
        self.assertIsNotNone(intArray2d)
        self.assertEqual(intArray2d["cType"],"int")
        self.assertEqual(intArray2d["rank"],2)
        t = next((var for var in test1["variables"] if var["name"] == "t".lower()),None)
        self.assertIsNotNone(t)
    def test_6_indexer_check_module_simple(self):
        simple = next((mod for mod in self._index if mod["name"] == "simple"),None)
        a = next((var for var in simple["variables"] if var["name"] == "a".lower()),None)
        self.assertIsNotNone(a)
        self.assertEqual(a["cType"],"int")
        self.assertEqual(a["rank"],0)
        n = next((var for var in simple["variables"] if var["name"] == "n".lower()),None)
        self.assertIsNotNone(n)
        self.assertEqual(n["cType"],"int")
        self.assertEqual(n["rank"],0)
        self.assertTrue(n["parameter"])
        self.assertEqual(n["value"].replace("(","").replace(")",""),"100")
        c = next((var for var in simple["variables"] if var["name"] == "c".lower()),None)
        self.assertIsNotNone(c)
        self.assertEqual(c["cType"],"float")
        self.assertEqual(c["rank"],2)
        self.assertTrue(c["device"])
        self.assertEqual(c["declareOnTarget"],"alloc")
        mytype = next((typ for typ in simple["types"] if typ["name"] == "mytype".lower()),None)
        self.assertIsNotNone(mytype)
        b = next((var for var in mytype["variables"] if var["name"] == "b".lower()),None) # mytype#b
        self.assertIsNotNone(b)
        self.assertEqual(b["cType"],"float")
        self.assertEqual(b["rank"],1)
        self.assertEqual(b["counts"][0],"n")
        self.assertEqual(b["lbounds"][0],"1")
    def test_7_indexer_check_module_nested_subprograms(self):
        nested_subprograms = next((mod for mod in self._index if mod["name"] == "nested_subprograms"),None)
        a = next((var for var in nested_subprograms["variables"] if var["name"] == "a".lower()),None)
        self.assertIsNotNone(a)
        self.assertEqual(a["cType"],"int")
        self.assertEqual(a["rank"],0)
        n = next((var for var in nested_subprograms["variables"] if var["name"] == "n".lower()),None)
        self.assertIsNotNone(n)
        self.assertEqual(n["cType"],"int")
        self.assertEqual(n["rank"],0)
        self.assertTrue(n["parameter"])
        self.assertEqual(n["value"].replace("(","").replace(")",""),"1000")
        e = next((var for var in nested_subprograms["variables"] if var["name"] == "e".lower()),None)
        self.assertIsNotNone(e)
        self.assertEqual(e["cType"],"float")
        self.assertEqual(e["rank"],2)
        self.assertEqual(e["counts"][0].replace(" ",""),"n-(-n)+1")
        self.assertEqual(e["counts"][1].replace(" ",""),"n-(-n)+1")
        self.assertEqual(e["lbounds"][0],"-n")
        self.assertEqual(e["lbounds"][1],"-n")
        self.assertFalse(e["device"])
        self.assertFalse(e["declareOnTarget"])
        mytype = next((typ for typ in nested_subprograms["types"] if typ["name"] == "mytype".lower()),None)
        self.assertIsNotNone(mytype)
        b = next((var for var in mytype["variables"] if var["name"] == "b".lower()),None) # mytype#b
        self.assertIsNotNone(b)
        self.assertEqual(b["cType"],"double")
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
        self.assertEqual(func2["resultName"],"res")
        self.assertEqual(len(func2["subprograms"]),1)
        # nested subprogram    
        func3 = next((sub for sub in func2["subprograms"] if sub["name"] == "func3".lower()),None)
        self.assertEqual(func3["kind"],"function")
        self.assertEqual(func3["resultName"],"func3")
        self.assertEqual(len(func3["subprograms"]),0)
        
      
if __name__ == '__main__':
    unittest.main() 
