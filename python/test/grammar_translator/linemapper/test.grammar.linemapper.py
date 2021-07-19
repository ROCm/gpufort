#!/usr/bin/env python3
import time
import unittest
import cProfile,pstats,io
import re
import ast

import addtoplevelpath
import linemapper.grammar as grammar
import linemapper.linemapper as linemapper

#import utils.logging
#
#LOG_FORMAT = "[%(levelname)s]\tgpufort:%(message)s"
#utils.logging.VERBOSE    = False

ENABLE_PROFILING = False

class TestPreprocessorGrammar(unittest.TestCase):
    def setUp(self):
        self._started_at = time.time()
        self._extra      = ""
        if ENABLE_PROFILING:
            self._profiler = cProfile.Profile()
            self._profiler.enable()
    def tearDown(self):
        if ENABLE_PROFILING:
            self._profiler.disable() 
            s = io.StringIO()
            sortby = 'cumulative'
            stats = pstats.Stats(self._profiler, stream=s).sort_stats(sortby)
            stats.print_stats(10)
            print(s.getvalue())
        elapsed = time.time() - self._started_at
        print('{} ({}s{})'.format(self.id(), round(elapsed, 6),self._extra))
    def test_1_compiler__options(self):
        options = "-Dflag0 -Dflag1=value1 -Dflag2=1 -Dflag3=.TRUE. -Dflag4=1.012E-23"
        names   = ["flag"+str(i) for i in range(0,4+1)]
        values  = [None, "value1", "1", ".TRUE.", "1.012E-23"]
        n = 0
        for result,_,__ in grammar.pp_compiler_option.scanString(options):
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
            for result,_,__ in grammar.pp_dir_undef.scanString(text):
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
            for result,_,__ in grammar.pp_dir_define.scanString(text):
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
            for result,_,__ in grammar.pp_dir_ifdef.scanString(text):
                self.assertEqual("a",result.name) 
        testdata = [
            "#ifndef a",
            "#      ifndef a",
        ]
        for text in testdata:
            for result,_,__ in grammar.pp_dir_ifndef.scanString(text):
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
            for result,_,__ in grammar.pp_dir_include.scanString(text):
                self.assertEqual(files[n],result.filename)
            n += 1
    def test_5_arithm_logic_expr(self):
        numSuccess = 0
        # basic checks
        testdata = [
            "a",
            "!a",
        ]
        binops      = ["==","!=","<",">","<=",">=","*","/","%","&&","||",".and.",".or."]
        expressions = ["a","1","0",".true.","a(b,c,d)"]
        for op in binops:
            for a in expressions:
                for b in expressions:
                    testdata.append("{} {} {}".format(a,op,b))
        numTests = len(testdata)
        for text in testdata:
            try:
                result = grammar.pp_arithm_logic_expr.parseString(text,parseAll=True)
                numSuccess += 1
            except:
                pass
        # some more complicated checks
        testdata.clear()
        testdata = [
          "( a == b ) || ( ( a*b <= 4 ) && defined(x) )",
          "defined(a) && macro(a*b+c,c+d-e%5) <= N%3",
          "defined(a) && macro1(macro2(a*b+macro3(c)),c+d-e%5) <= N%3",
        ]
        numTests += len(testdata)
        for text in testdata:
            try:
                result = grammar.pp_arithm_logic_expr.parseString(text,parseAll=True)
                numSuccess += 1
            except Exception as e:
                print(text)
                raise e
        self._extra = ", performed {} checks".format(numTests)
        self.assertEqual(numSuccess,numTests)
    def test_6_macro_substitution(self):
        macroStack = [
          { "name": "b", "args": [], "subst": "5" },
          { "name": "a", "args": ["x"], "subst": "(5*x)" },
        ]
        
        testdata_true = [
          "defined(a)",
          "!!defined(a)",
          "!defined(x)",
          "defined(a) && !defined(x) && a(b) > 4"
        ]
        testdata_false = [
          "!defined(a)",
          "!!defined(x)",
          "!defined(a) || defined(x) || a(5) < 1",
          "!(defined(a) && !defined(x) && a(b) > 4)"
        ]
        for text in testdata_true:
            condition = linemapper.evaluateCondition(text,macroStack)
            self.assertTrue(condition)
        for text in testdata_false:
            condition = linemapper.evaluateCondition(text,macroStack)
            self.assertFalse(condition)
        numTests = len(testdata_true) + len(testdata_false)
        self._extra = ", performed {} checks".format(numTests)
      
if __name__ == '__main__':
    unittest.main() 