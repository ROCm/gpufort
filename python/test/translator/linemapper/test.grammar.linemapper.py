#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import time
import unittest
import cProfile,pstats,io
import re
import ast

import addtoplevelpath
from gpufort import linemapper

#import gpufort.util.logging
#
#LOG_FORMAT = "[%(levelname)s]\tgpufort:%(message)s"
#gpufort.util.logging.opts.verbose    = False

PROFILING_ENABLE = False

class TestPreprocessorGrammar(unittest.TestCase):
    def setUp(self):
        self.started_at = time.time()
        self.extra      = ""
        if PROFILING_ENABLE:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
    def tearDown(self):
        if PROFILING_ENABLE:
            self.profiler.disable() 
            s = io.StringIO()
            sortby = 'cumulative'
            stats = pstats.Stats(self.profiler, stream=s).sort_stats(sortby)
            stats.print_stats(10)
            print(s.getvalue())
        elapsed = time.time() - self.started_at
        print('{} ({}s{})'.format(self.id(), round(elapsed, 6),self.extra))
    def test_1_compiler__options(self):
        options = "-Dflag0 -Dflag1=value1 -Dflag2=1 -Dflag3=.TRUE. -Dflag4=1.012E-23"
        names   = ["flag"+str(i) for i in range(0,4+1)]
        values  = [None, "value1", "1", ".TRUE.", "1.012E-23"]
        n = 0
        for result,_,__ in linemapper.grammar.pp_compiler_option.scanString(options):
            self.assertEqual(values[n],result.value,"Parsing value for name '"+result.name+"' failed") 
            n += 1
    def test_2_definitions(self):
        #undef
        testdata = [
            "#undef a",
            "#undef a1 5",
            "# undef _a_b_c 6",
        ]
        names = ["a","a1","_a_b_c","a","a"]
        n = 0
        for text in testdata:
            for result,_,__ in linemapper.grammar.pp_dir_undef.scanString(text):
                self.assertEqual(names[n],result.name) 
            n += 1
        #define
        testdata = [
            "#define a",
            "#define a1 5",
            "# define _a_b_c 6",
            "#define a text",
            "#define a(b,c,d) (b)*(c)*(d)",
        ]
        args          = [None, None, None, None, ["b","c","d"]]
        substitutions = ["", "5", "6", "text", "(b)*(c)*(d)"]
        n = 0
        for text in testdata:
            for result,_,__ in linemapper.grammar.pp_dir_define.scanString(text):
                self.assertEqual(names[n],result.name) 
                if result.args:
                    result_args = result.args.replace(" ","").split(",")
                    self.assertEqual(args[n],result_args) 
                else:
                    self.assertEqual(args[n],result.args) 
                self.assertEqual(substitutions[n],result.subst) 
            n += 1
    def test_3_simple_conditionals(self):
        testdata = [
            "#ifdef a",
            "#      ifdef a",
        ]
        for text in testdata:
            for result,_,__ in linemapper.grammar.pp_dir_ifdef.scanString(text):
                self.assertEqual("a",result.name) 
        testdata = [
            "#ifndef a",
            "#      ifndef a",
        ]
        for text in testdata:
            for result,_,__ in linemapper.grammar.pp_dir_ifndef.scanString(text):
                self.assertEqual("a",result.name) 
    def test_4_include(self):
        files = [
            "myfile.f90",
            "mypath/myfile.f90",
            "/home/user/mypath/myfile.f90",
            "./mypath/myfile.f90",
            "../mypath/myfile.f90",
            "../mypath/myfile.f90",
        ]
        testdata = [
            "#include \"myfile.f90\"",
            "#include \"mypath/myfile.f90\"",
            "#include \"/home/user/mypath/myfile.f90\"",
            "#include \"./mypath/myfile.f90\"",
            "#include \"../mypath/myfile.f90\"",
            "#   include \"../mypath/myfile.f90\"",
        ]
        n = 0
        for text in testdata:
            for result,_,__ in linemapper.grammar.pp_dir_include.scanString(text):
                self.assertEqual(files[n],result.filename)
            n += 1
      
if __name__ == '__main__':
    unittest.main() 