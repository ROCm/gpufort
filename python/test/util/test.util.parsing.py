#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import time
import unittest
import cProfile,pstats,io
import json

import addtoplevelpath
from gpufort import util

LOG_FORMAT = "[%(levelname)s]\tgpufort:%(message)s"
util.logging.opts.verbose    = False
util.logging.init_logging("log.log",LOG_FORMAT,"warning")

PROFILING_ENABLE = False

# whitespace at begin is important to
# as [cC] in first column indicates a comment
# line in Fortran 77.
testdata1 = """\
    call myroutine( & ! comment 1
        arg1,&
        ! comment 2


        arg2) ! comment 3
    !$acc directive & ! comment 4
    !$acc clause1(var) &
    !$acc& clause2(var)\
"""

testdata1_result = """\
    call myroutine( &
        arg1,&
        arg2)
    ! comment 1
        ! comment 2
        ! comment 3
    !$acc directive &
    !$acc clause1(var) &
    !$acc& clause2(var)
    ! comment 4\
"""

testdata3="k () + a ( b, c(d)+e(f)) + g(h(i,j+a(k(),2)))"""

testdata3_result= {
    "a" :  [('a ( b, c(d)+e(f))', [' b', ' c(d)+e(f)']), ('a(k(),2)', ['k()', '2'])],
    "b" :  [],
    "c" :  [('c(d)', ['d'])],
    "d" :  [],
    "f" :  [],
    "g" :  [('g(h(i,j+a(k(),2)))', ['h(i,j+a(k(),2))'])],
    "h" :  [('h(i,j+a(k(),2))', ['i', 'j+a(k(),2)'])],
    "i" :  [],
    "j" :  [],
    "k" :  [('k ()', []), ('k()', [])],
    }


class TestParsingUtils(unittest.TestCase):
    def prepare(self,text):
        return text.strip().splitlines()
    def clean(self,text):
        return text.replace(" ","").replace("\t","").replace("\n","").replace("\r","")
    def setUp(self):
        global PROFILING_ENABLE
        if PROFILING_ENABLE:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
        self.started_at = time.time()
    def tearDown(self):
        global PROFILING_ENABLE
        if PROFILING_ENABLE:
            self.profiler.disable() 
            s = io.StringIO()
            sortby = 'cumulative'
            stats = pstats.Stats(self.profiler, stream=s).sort_stats(sortby)
            stats.print_stats(10)
            print(s.getvalue())
        elapsed = time.time() - self.started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 6)))
    def test_00_tokenize(self):
        statements = [
          "a,b(i,j),c(i,j,k)",  # 17 tokens
          "call debug(1, \"my literal string with ; and \\\"; use something \")",
          "call debug(1, 'my literal string with ; and \\'; use something ')",
          "end do",
          "enddo",
          "end if",
          "endif",
          "else if",
          "elseif",
          "!$acc enter data copyin(a) copyout(b(-1:))",
        ]
        results = [
          ['a', ',', 'b', '(', 'i', ',', 'j', ')', ',', 'c', '(', 'i', ',', 'j', ',', 'k', ')'],
          ['call', 'debug', '(', '1', ',', '"my literal string with ; and \\"; use something "', ')'],
          ['call', 'debug', '(', '1', ',', "'my literal string with ; and \\'; use something '", ')'],
          ['end', 'do'],
          ['enddo'],
          ['end', 'if'],
          ['endif'],
          ['else', 'if'],
          ['elseif'],
          ['!$', 'acc', 'enter', 'data', 'copyin', '(', 'a', ')', 'copyout', '(', 'b', '(', '-', '1', ':', ')', ')'],
        ]
        for i,stmt in enumerate(statements):
            #print(util.parsing.tokenize(stmt))
            self.assertEqual(util.parsing.tokenize(stmt),results[i])
    def test_01_next_tokens_till_open_bracket_is_closed(self):
        expressions = [
         "a,b,c) f(g,h)",  
         "a,b,c) f(g,h)",  
         "(a,b,c))) f(g,h)",  
         "(a,b,c))) f(g,h)",  
         "integer (a,b,c)",  
         "integer(kind=8) (a,b,c)",  
        ]
        params = [
          #(open_brackets,brackets,keepend)
          (1,("(",")"),True),
          (1,("(",")"),False),
          (2,("(",")"),True),
          (2,("(",")"),False),
          (1,(None,")"),True),
          (1,(None,")"),True),
        ] 
        results = [
          ["a",",","b",",","c",")"],
          ["a",",","b",",","c"],
          ["(","a",",","b",",","c",")",")",")"],
          ["(","a",",","b",",","c",")",")",],
          ['integer', '(', 'a', ',', 'b', ',', 'c', ')'],
          ['integer', '(', 'kind', '=', '8', ')'],
        ]
        for i,stmt in enumerate(expressions):
            result = util.parsing.next_tokens_till_open_bracket_is_closed(util.parsing.tokenize(stmt),
                                                                       params[i][0],
                                                                       params[i][1],
                                                                       params[i][2])
            #print(result)
            self.assertEqual(result,results[i])
    def test_02_split_fortran_line(self):
        testdata = \
"""
! comment
  ! comment
stmt_or_dir ! comment
!$acc stmt_or_dir
*$acc stmt_or_dir
c$acc stmt_or_dir
C$acc stmt_or_dir
!$ acc stmt_or_dir
! $acc comment
  !a$acc comment
500 stmt_or_dir
"""

        for line in self.prepare(testdata):
            result = util.parsing.split_fortran_line(line)
            numeric_label_part,indent,\
            stmt_or_dir,comment,trailing_ws =\
              result
            if "stmt_or_dir" in line:
                self.assertTrue(len(stmt_or_dir))
            else:
                self.assertFalse(len(stmt_or_dir))
            if "comment" in line:
                self.assertTrue(len(comment))
            else:
                self.assertFalse(len(comment))
            if "500" in line:
                self.assertTrue(len(numeric_label_part))
            else:
                self.assertFalse(len(numeric_label_part))
        # 
    def test_03_detect_line_starts(self):
        input_lines = testdata1.splitlines()
        linenos = util.parsing.detect_line_starts(\
                   input_lines)
        #print(linenos)
        #print([a for i,a in enumerate(input_lines) if i in linenos]) 
        self.assertEqual(linenos,[0,6,9])
    def test_04_relocate_inline_comments(self):
        result = util.parsing.relocate_inline_comments(\
                   testdata1.splitlines())
        self.assertEqual(self.clean("\n".join(result)),self.clean(testdata1_result))
    def test_05_get_top_level_operands(self):
        statements = [
          "a,b(i,j),c(i,j,k)",  # 17 tokens
          "a,b(i,j),c(i,j,k))",
          "a,b(i,j),c(i,j,k)))",
          "-1:2*(m+b)*c:k", # 14 tokens
          "a%b(i%a(5)%i3(mm%i4),j)%c(i%k%n,j,k%k%j)",
          "a%b(a,:) = myfunc(a,f=3)*b(:)",
          #"integer * 5 pure recursive function foo( arg1, arg2, arg3 ) result(i5) bind(c,name=\"cname\")",
        ]
        separators = [
          [","],
          [","],
          [","],
          [":"],
          ["%"],
          ["="],
        ]
        results = [
          (['a', 'b(i,j)', 'c(i,j,k)'],17),
          (['a', 'b(i,j)', 'c(i,j,k)'],17),
          (['a', 'b(i,j)', 'c(i,j,k)'],17),
          (['-1', '2*(m+b)*c', 'k'],14),
          (['a', 'b(i%a(5)%i3(mm%i4),j)', 'c(i%k%n,j,k%k%j)'], 37),
          (['a%b(a,:)', 'myfunc(a,f=3)*b(:)'], 22),
        ]
        for i,stmt in enumerate(statements):
           #print(util.parsing.get_top_level_operands(util.parsing.tokenize(stmt),
           #                                          separators=separators[i]))
           self.assertEqual(util.parsing.get_top_level_operands(util.parsing.tokenize(stmt),
                            separators=separators[i]),results[i])

    def test_06_extract_function_calls(self):
        for c in ["a","b","c","d","f","g","h","i","j","k"]:
        #for c in ["a"]:
            result = util.parsing.extract_function_calls(testdata3,c)
            #print(result)
            self.assertEqual(result,testdata3_result[c])
    
    def test_07_strip_array_indexing(self):
        expressions = [
          "a",
          "a(1)",
          "a(:,5)",
          "a(:,5)%b",
          "A(:,5)%b(c(5,2))%c",
        ]
        results = [
          "a",
          "a",
          "a",
          "a%b",
          "A%b%c",
        ]
        for i,expr in enumerate(expressions):
            #print(util.parsing.strip_array_indexing(expr))
            self.assertEqual(util.parsing.strip_array_indexing(expr),results[i])
    
    def test_08_derived_type_parents(self):
        expressions = [
          "a",
          "a(1)",
          "a(:,5)",
          "a(:,5)%b",
          "A(:,5)%b(c(5,2))%c",
        ]
        results = [
          [],
          [],
          [],
          ['a'],
          ['A', 'A%b'],
        ]
        for i,expr in enumerate(expressions):
            #print(util.parsing.derived_type_parents(expr))
            self.assertEqual(util.parsing.derived_type_parents(expr),results[i])
   
    def test_09_mangle_fortran_var_expr(self):
        expressions = [
          "a(i,j)%b%arg3(1:n)",
        ]
        results = [
          "aLijR_b_arg3L1TnR",
        ]
        for i,expr in enumerate(expressions):
            #print(util.parsing.mangle_fortran_var_expr(expr))
            self.assertEqual(util.parsing.mangle_fortran_var_expr(expr),results[i])
    
    def test_10_expand_letter_range(self):
        expressions = [
         "a",
         "a-a",
         "a-d", 
         "g-h",
        ]
        results = [
          ["a"],
          ["a"],
          ["a","b","c","d"],
          ["g","h"],
        ]
        for i,expr in enumerate(expressions):
            self.assertEqual(util.parsing.expand_letter_range(expr),results[i])
    
if __name__ == '__main__':
    unittest.main() 
