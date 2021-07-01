#!/usr/bin/env python3
import time
import unittest
import cProfile,pstats,io
import re
import ast

import addtoplevelpath
import preprocessor.grammar as grammar

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
        args          = [[], [], [], [], ["b","c","d"]]
        substitutions = ["", "5", "6", "text", "(b)*(c)*(d)"]
        n = 0
        for text in testdata:
            for result,_,__ in grammar.pp_dir_define.scanString(text):
                self.assertEqual(names[n],result.name) 
                self.assertEqual(args[n],result.args.asList()) 
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
            except:
                print(text)
                pass
        self._extra = ", performed {} checks".format(numTests)
        self.assertEqual(numSuccess,numTests)
    def test_6_macro_substitution(self):
        macroStack = [
          { "name": 'b', "args": [], "subst": "5" },
          { "name": 'a', "args": ["x"], "subst": "(5*x)" },
        ]
        def defined_(tokens):
            nonlocal macroStack
            if next((macro for macro in macroStack if macro["name"] == tokens[0]),None):
                return "1"
            else: 
                return "0"
        def substitute_(tokens):
            name = tokens[0]
            args = tokens[1]
            nonlocal macroStack
            macro = next((macro for macro in reversed(macroStack) if macro["name"] == tokens[0]),None)
            if macro:
                subst = macro["subst"]
                for n,placeholder in enumerate(macro["args"]):
                    subst = re.sub(r"\b{}\b".format(placeholder),args[n],subst)      
                return subst
            else:
                tokens[0]
        def transformArithmLogicExpr_(original):
             oldResult = None
             result    = original
             # expand macro; one at a time
             while result != oldResult:
                   oldResult = result
                   result    = grammar.pp_value.transformString(result)
             # replace C and Fortran operatos by python equivalents 
             result = grammar.pp_ops.transformString(result)
             return result
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
        grammar.pp_defined.setParseAction(defined_)
        grammar.pp_macro_eval.setParseAction(substitute_)
        for text in testdata_true:
            result    = transformArithmLogicExpr_(text)
            condition = bool(eval(result))
            self.assertTrue(condition)
        for text in testdata_false:
            result = transformArithmLogicExpr_(text)
            condition = bool(eval(result))
            self.assertFalse(condition)
      
if __name__ == '__main__':
    unittest.main() 
