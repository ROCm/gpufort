#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os,sys
import time
import unittest
import addtoplevelpath
import inspect

from gpufort import translator

print("Running test '{}'".format(os.path.basename(__file__)),end="",file=sys.stderr)

def tostr(opt_single_val):
    return translator.tree.traversals.make_fstr(
      opt_single_val.value
    ) if opt_single_val.specified else None
        
device_type = "radeon"

testdata_acc_constructs_no_clauses = [
  "!$acc serial",
  "!$acc parallel",
  "!$acc kernels",
  "!$acc parallel loop",
  "!$acc kernels loop",
]       

testdata_acc_loops_no_clauses = [
  "!$acc loop",
]

testdata_cuf_construct_no_clauses = [
  "!$cuf kernels do",
]       

pyparsing_expr = (
  translator.tree.grammar.loop_annotation
  | translator.tree.grammar.offload_region_start
)

class TestAnalysisAcc(unittest.TestCase):
    def setUp(self):
        global index
        self.started_at = time.time()
    def tearDown(self):
        elapsed = time.time() - self.started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 9)))
    def test_00_do_nothing(self):
        pass
    def test_01_parse_acc_constructs_with_no_clauses(self):
        for i,test in enumerate(testdata_acc_constructs_no_clauses):
            pyparsing_expr.parseString(test,parseAll=True)
        for i,test in enumerate(testdata_acc_loops_no_clauses):
            pyparsing_expr.parseString(test,parseAll=True)
    def test_02_analyze_acc_constructs_with_no_clauses(self):
        for i,test in enumerate(testdata_acc_constructs_no_clauses):
            parse_result = pyparsing_expr.parseString(test,parseAll=True)[0]
            translator.analysis.acc.analyze_directive(
              parse_result,
              device_type 
            )
        for i,test in enumerate(testdata_acc_loops_no_clauses):
            parse_result = pyparsing_expr.parseString(test,parseAll=True)[0]
            translator.analysis.acc.analyze_directive(
              parse_result,
              device_type 
            )
    def test_03_analyze_acc_loop_no_values(self):
        testdata_acc_loops_no_values = [
          "!$acc loop",
          "!$acc loop gang",
          "!$acc loop worker",
          "!$acc loop vector",
          "!$acc loop gang worker",
          "!$acc loop gang vector",
          "!$acc loop gang worker vector",
        ]
        results = [
          [False,False,False], 
          [True,False,False], 
          [False,True,False], 
          [False,False,True], 
          [True,True,False], 
          [True,False,True], 
          [True,True,True], 
        ]
        for i,test in enumerate(testdata_acc_loops_no_values):
            parse_result = pyparsing_expr.parseString(test,parseAll=True)[0]
            acc_loop_info = translator.analysis.acc.analyze_directive(
              parse_result,
              device_type 
            )
            self.assertEqual(
              [
                acc_loop_info.gang.specified,
                acc_loop_info.worker.specified,
                acc_loop_info.vector.specified,
              ],
              results[i]
            )
    def test_04_analyze_acc_loop_device_types_no_values(self):
        testdata_acc_loops_no_values = [
          "!$acc loop device_type(radeon) gang",
          "!$acc loop device_type(radeon) worker",
          "!$acc loop device_type(radeon) vector",
          "!$acc loop device_type(nvidia) gang worker",
          "!$acc loop device_type(nvidia) gang vector",
          "!$acc loop device_type(nvidia) gang worker vector",
          "!$acc loop dtype(radeon) gang",
          "!$acc loop dtype(radeon) worker",
          "!$acc loop dtype(radeon) vector",
          "!$acc loop dtype(nvidia) gang worker",
          "!$acc loop dtype(nvidia) gang vector",
          "!$acc loop dtype(nvidia) gang worker vector",
          "!$acc loop device_type(*) vector dtype(nvidia) gang worker",
          "!$acc loop device_type(*) vector dtype(nvidia) gang worker",
          "!$acc loop device_type(*) vector dtype(nvidia) gang worker",
          "!$acc loop gang device_type(*) vector dtype(nvidia) worker",
          "!$acc loop gang device_type(*) vector dtype(nvidia) worker",
          "!$acc loop gang device_type(*) vector dtype(nvidia) worker",
        ]
        results_radeon = [
          [True,False,False], 
          [False,True,False], 
          [False,False,True], 
          [False,False,False],
          [False,False,False],
          [False,False,False],
          [True,False,False], 
          [False,True,False], 
          [False,False,True], 
          [False,False,False],
          [False,False,False],
          [False,False,False],
          [False,False,True], 
          [False,False,True], 
          [False,False,True], 
          [True,False,True], 
          [True,False,True], 
          [True,False,True], 
        ]
        results_nvidia = [
          [False,False,False],
          [False,False,False],
          [False,False,False],
          [True,True,False], 
          [True,False,True], 
          [True,True,True], 
          [False,False,False],
          [False,False,False],
          [False,False,False],
          [True,True,False], 
          [True,False,True], 
          [True,True,True], 
          [True,True,False], 
          [True,True,False], 
          [True,True,False], 
          [True,True,False], 
          [True,True,False], 
          [True,True,False], 
        ]
        results = {
          "radeon": results_radeon,
          "nvidia": results_nvidia,
        }
        for device_type in ["radeon","nvidia"]:
            #print(device_type)
            for i,test in enumerate(testdata_acc_loops_no_values):
                parse_result = pyparsing_expr.parseString(test,parseAll=True)[0]
                acc_loop_info = translator.analysis.acc.analyze_directive(
                  parse_result,
                  device_type 
                )
                #print(test)
                self.assertEqual(
                  [
                    acc_loop_info.gang.specified,
                    acc_loop_info.worker.specified,
                    acc_loop_info.vector.specified,
                  ],
                  results[device_type][i]
                )
    def test_05_analyze_acc_loop_values(self):
        testdata = [
          "!$acc loop",
          "!$acc loop gang(2) worker(4) vector(8) collapse(2)",
          "!$acc loop gang(2) worker(4) vector(8) tile(TILE_SIZE,32)",
        ]
        results = [
          [False,False,False,False,False,
           None,None,None,None,[]], 
          [True,True,True,True,False,
           "2","4","8","2",[]], 
          [True,True,True,False,True,
           "2","4","8",None,["TILE_SIZE","32"]], 
        ]
        for i,test in enumerate(testdata):
            parse_result = pyparsing_expr.parseString(test,parseAll=True)[0]
            acc_loop_info = translator.analysis.acc.analyze_directive(
              parse_result,
              device_type 
            )
            #print(test)
            self.assertEqual(
              [
                acc_loop_info.gang.specified,
                acc_loop_info.worker.specified,
                acc_loop_info.vector.specified,
                acc_loop_info.collapse.specified,
                acc_loop_info.tile.specified,
                tostr(acc_loop_info.gang),
                tostr(acc_loop_info.worker),
                tostr(acc_loop_info.vector),
                tostr(acc_loop_info.collapse),
                [translator.tree.traversals.make_fstr(e)
                for e in acc_loop_info.tile.value],
              ],
              results[i]
            )
        pass
if __name__ == '__main__':
    unittest.main() 
