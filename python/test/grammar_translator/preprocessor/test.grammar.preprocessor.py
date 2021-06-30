#!/usr/bin/env python3
import time
import unittest
import cProfile,pstats,io

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
        print('{} ({}s)'.format(self.id(), round(elapsed, 6)))
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
        testdata = [
            "a",
            "!a",
        ]
        binops      = ["==","!=","<",">","<=",">=","*","/","%","&&","||","and"
        expressions = ["a","1","0",".true.",".false.","true","false","a(b,c,d)"]
        for expressions in 
            testdata.append(

        n = 0
        for text in testdata:
            for result,_,__ in grammar.pp_arithm_logic_expr.scanString(text):
                print(result)
            n += 1

      
if __name__ == '__main__':
    unittest.main() 
