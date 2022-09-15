#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os,sys
import time
import unittest
import copy

# TODO use standard way after debugging
#import addtoplevelpath
#from gpufort import translator
import loop_transformations

print("Running test '{}'".format(os.path.basename(__file__)),end="",file=sys.stderr)
        
# 0 levels
testdata_seq = [
  loop_transformations.Loop("i","0",length="N"),
  loop_transformations.Loop("i","0",last="last"),
  loop_transformations.Loop("i","0",excl_ubound="excl_ubound"),
  loop_transformations.Loop("i","first",length="N"),
  loop_transformations.Loop("i","first",last="last"),
  loop_transformations.Loop("i","first",excl_ubound="excl_ubound"),
  loop_transformations.Loop("i","0",length="N",step="2"),
  loop_transformations.Loop("i","0",last="last",step="2"),
  loop_transformations.Loop("i","0",excl_ubound="excl_ubound",step="2"),
  loop_transformations.Loop("i","first",length="N",step="2"),
  loop_transformations.Loop("i","first",last="last",step="2"),
  loop_transformations.Loop("i","first",excl_ubound="excl_ubound",step="2"),
]
# 1 level (max)
testdata_gang_max = copy.deepcopy(testdata_seq)
for loop in testdata_gang_max:
    loop.gang_partitioned=True
testdata_worker_max = copy.deepcopy(testdata_seq)
for loop in testdata_worker_max:
    loop.worker_partitioned=True
testdata_vector_max = copy.deepcopy(testdata_seq)
for loop in testdata_vector_max:
    loop.vector_partitioned=True
# 2 levels (max,max)
testdata_gang_max_worker_max = copy.deepcopy(testdata_seq)
for loop in testdata_gang_max_worker_max:
    loop.gang_partitioned=True
    loop.worker_partitioned=True
testdata_gang_max_vector_max = copy.deepcopy(testdata_seq)
for loop in testdata_gang_max_vector_max:
    loop.gang_partitioned=True
    loop.vector_partitioned=True
# 3 levels (max,max,max)
testdata_gang_max_worker_max_vector_max = copy.deepcopy(testdata_seq)
for loop in testdata_gang_max_worker_max_vector_max:
    loop.gang_partitioned=True
    loop.worker_partitioned=True
    loop.vector_partitioned=True

class TestLoopTransformations(unittest.TestCase):
    def clean(self,text):
        if text!=None:
            return text.replace(" ","").replace("\t","").replace("\n","").replace("\r","")
        else:
            return None
    def setUp(self):
        global index
        self.started_at = time.time()
        loop_transformations.reset()
    def tearDown(self):
        elapsed = time.time() - self.started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 9)))
    def test_00_do_nothing(self):
        pass
    def test_01_map_seq_loop_to_hip_cpp(self):
        results = [
          ('for (i = 0; \n     i < N; i++) {\n', '}\n', None),
          ('for (i = 0; \n     i < (last + 1); i++) {\n', '}\n', None),
          ('for (i = 0; \n     i < excl_ubound; i++) {\n', '}\n', None),
          ('for (i = first; \n     i < (first + N); i++) {\n', '}\n', None),
          ('for (i = first; \n     i < (last + 1); i++) {\n', '}\n', None),
          ('for (i = first; \n     i < excl_ubound; i++) {\n', '}\n', None),
          ('for (i = 0;\n     i < (2)*N; i += 2) {\n', '}\n', None),
          ('for (i = 0;\n     i < (last + 1); i += 2) {\n', '}\n', None),
          ('for (i = 0;\n     i < excl_ubound; i += 2) {\n', '}\n', None),
          ('for (i = first;\n     i < (first + (2)*N); i += 2) {\n', '}\n', None),
          ('for (i = first;\n     i < (last + 1); i += 2) {\n', '}\n', None),
          ('for (i = first;\n     i < excl_ubound; i += 2) {\n', '}\n', None),
        ]
        for i,loop in enumerate(testdata_seq):
            result = loop.map_to_hip_cpp() 
            self.assertEqual(self.clean(result[0]),self.clean(results[i][0]))
            #print(str(result)+",")
    def test_02_tile_seq_loop_and_map_to_hip_cpp(self):
        results = [
        ]
        for loop in testdata_seq:
            tile_loop, element_loop = loop.tile("tile_size")
            tile_loop_result = tile_loop.map_to_hip_cpp()[0] 
            element_loop_result = element_loop.map_to_hip_cpp()[0] 
            #print(str(tile_loop_result[0]),end="")
            #print(str(element_loop_result[0])+",")
    def test_03_map_vector_max_loop_to_hip_cpp(self):
        results = [
        ]
        for loop in testdata_vector_max:
            prolog,epilog,statement_activation_cond,indent = loop.map_to_hip_cpp() 
            result = prolog
            result += "{}if ( {} ) {{ /*...*/ }}\n".format(indent,statement_activation_cond)
            result += epilog
            #print(result)
    def test_04_map_worker_max_loop_to_hip_cpp(self):
        results = [
        ]
        for loop in testdata_worker_max:
            prolog,epilog,statement_activation_cond,indent = loop.map_to_hip_cpp() 
            result = prolog
            result += "{}if ( {} ) {{ /*...*/ }}\n".format(indent,statement_activation_cond)
            result += epilog
            #print(result)
    def test_05_map_gang_max_loop_to_hip_cpp(self):
        results = [
        ]
        for loop in testdata_gang_max:
            prolog,epilog,statement_activation_cond,indent = loop.map_to_hip_cpp() 
            result = prolog
            result += "{}if ( {} ) {{ /*...*/ }}\n".format(indent,statement_activation_cond)
            result += epilog
            #print(result)
    def test_06_map_gang_max_worker_max_loop_to_hip_cpp(self):
        results = [
        ]
        for loop in testdata_gang_max_worker_max:
            prolog,epilog,statement_activation_cond,indent = loop.map_to_hip_cpp() 
            result = prolog
            result += "{}if ( {} ) {{ /*...*/ }}\n".format(indent,statement_activation_cond)
            result += epilog
            #print(result)
    def test_07_map_gang_max_worker_max_vector_max_loop_to_hip_cpp(self):
        results = [
        ]
        for loop in testdata_gang_max_worker_max_vector_max:
            prolog,epilog,statement_activation_cond,indent = loop.map_to_hip_cpp() 
            result = prolog
            result += "{}if ( {} ) {{ /*...*/ }}\n".format(indent,statement_activation_cond)
            result += epilog
            #print(result)

testdata_loopnest_seq = [
  [  
    loop_transformations.Loop("i","i_first",last="i_last",step="i_step"),
    loop_transformations.Loop("j","j_first",last="j_last",step="j_step"),
    loop_transformations.Loop("k","k_first",last="k_last",step="k_step"),
  ],
]

# 1 level (max)
testdata_loopnest_gang_max = copy.deepcopy(testdata_loopnest_seq)
for loops in testdata_loopnest_gang_max:
    loops[0].gang_partitioned = True
testdata_loopnest_worker_max = copy.deepcopy(testdata_loopnest_seq)
for loops in testdata_loopnest_worker_max:
    loops[0].worker_partitioned = True
testdata_loopnest_vector_max = copy.deepcopy(testdata_loopnest_seq)
for loops in testdata_loopnest_vector_max:
    loops[0].vector_partitioned = True
# 2 levels (max,max)
testdata_loopnest_gang_max_worker_max = copy.deepcopy(testdata_loopnest_seq)
for loops in testdata_loopnest_gang_max_worker_max:
    loops[0].gang_partitioned = True
    loops[0].worker_partitioned = True
testdata_loopnest_gang_max_vector_max = copy.deepcopy(testdata_loopnest_seq)
for loops in testdata_loopnest_gang_max_vector_max:
    loops[0].gang_partitioned = True
    loops[0].vector_partitioned = True
testdata_loopnest_worker_max_vector_max = copy.deepcopy(testdata_loopnest_seq)
for loops in testdata_loopnest_worker_max_vector_max:
    loops[0].worker_partitioned = True
    loops[0].vector_partitioned = True
# 3 levels (max,max,max)
testdata_loopnest_gang_max_worker_max_vector_max = copy.deepcopy(testdata_loopnest_seq)
for loops in testdata_loopnest_gang_max_worker_max_vector_max:
    loops[0].gang_partitioned = True
    loops[0].worker_partitioned = True
    loops[0].vector_partitioned = True
 
class TestLoopnestTransformations(unittest.TestCase):
    def clean(self,text):
        if text!=None:
            return text.replace(" ","").replace("\t","").replace("\n","").replace("\r","")
        else:
            return None
    def setUp(self):
        global index
        self.started_at = time.time()
        loop_transformations.reset()
    def tearDown(self):
        elapsed = time.time() - self.started_at
    def test_00_do_nothing(self):
        pass
    def test_01_collapse_loopnest_seq(self):
        for i,loops in enumerate(testdata_loopnest_seq):
            loopnest = loop_transformations.Loopnest(loops)
            collapsed_loop = loopnest.collapse()
            prolog,epilog,statement_activation_cond,indent = collapsed_loop.map_to_hip_cpp() 
            result = prolog
            #result += "{}if ( {} ) {{ /*...*/ }}\n".format(indent,statement_activation_cond)
            result += epilog
            #print(result)
    def test_02_collapse_loopnest_vector_max(self):
        for i,loops in enumerate(testdata_loopnest_vector_max):
            loopnest = loop_transformations.Loopnest(loops)
            collapsed_loop = loopnest.collapse()
            prolog,epilog,statement_activation_cond,indent = collapsed_loop.map_to_hip_cpp() 
            result = prolog
            #result += "{}if ( {} ) {{ /*...*/ }}\n".format(indent,statement_activation_cond)
            result += epilog
            #print(result)
    def test_03_collapse_loopnest_worker_max(self):
        for i,loops in enumerate(testdata_loopnest_worker_max):
            loopnest = loop_transformations.Loopnest(loops)
            collapsed_loop = loopnest.collapse()
            prolog,epilog,statement_activation_cond,indent = collapsed_loop.map_to_hip_cpp() 
            result = prolog
            #result += "{}if ( {} ) {{ /*...*/ }}\n".format(indent,statement_activation_cond)
            result += epilog
            #print(result)
    def test_04_collapse_loopnest_gang_max(self):
        for i,loops in enumerate(testdata_loopnest_gang_max):
            loopnest = loop_transformations.Loopnest(loops)
            collapsed_loop = loopnest.collapse()
            prolog,epilog,statement_activation_cond,indent = collapsed_loop.map_to_hip_cpp() 
            result = prolog
            #result += "{}if ( {} ) {{ /*...*/ }}\n".format(indent,statement_activation_cond)
            result += epilog
            #print(result)
    def test_05_tile_loopnest_collapse_tile_collapse_element_gang_max(self):
        for i,loops in enumerate(testdata_loopnest_gang_max):
            tile_sizes = ["32","8","LEN"]
            loopnest = loop_transformations.Loopnest(loops).tile(tile_sizes)
            prolog,epilog,statement_activation_cond,indent = loopnest.map_to_hip_cpp() 
            result = prolog
            #result += "{}if ( {} ) {{ /*...*/ }}\n".format(indent,statement_activation_cond)
            result += epilog
            print(result)
    def test_06_tile_loopnest_collapse_tile_collapse_element_worker_max(self):
        for i,loops in enumerate(testdata_loopnest_worker_max):
            tile_sizes = ["32","8","LEN"]
            loopnest = loop_transformations.Loopnest(loops).tile(tile_sizes)
            prolog,epilog,statement_activation_cond,indent = loopnest.map_to_hip_cpp() 
            result = prolog
            #result += "{}if ( {} ) {{ /*...*/ }}\n".format(indent,statement_activation_cond)
            result += epilog
            print(result)
    def test_07_tile_loopnest_collapse_tile_collapse_element_vector_max(self):
        for i,loops in enumerate(testdata_loopnest_vector_max):
            tile_sizes = ["32","8","LEN"]
            loopnest = loop_transformations.Loopnest(loops).tile(tile_sizes,True,True)
            prolog,epilog,statement_activation_cond,indent = loopnest.map_to_hip_cpp() 
            result = prolog
            #result += "{}if ( {} ) {{ /*...*/ }}\n".format(indent,statement_activation_cond)
            result += epilog
            print(result)
                 

if __name__ == '__main__':
    unittest.main() 
