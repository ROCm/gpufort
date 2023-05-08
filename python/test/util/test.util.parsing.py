#!/usr/bin/env python3

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

testdata1 = \
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
"""

# whitespace at begin is important to
# as [cC] in first column indicates a comment
# line in Fortran 77.
testdata2 = """\
    call myroutine( & ! comment 1
        arg1,&
        ! comment 2


        arg2) ! comment 3
    !$acc directive & ! comment 4
    !$acc clause1(var) &
    !$acc& clause2(var)\
"""

testdata2_result = """\
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
    def test_01_tokenize(self):
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
          ['end', 'do'],
          ['end', 'if'],
          ['end', 'if'],
          ['else', 'if'],
          ['else', 'if'],
          ['!$', 'acc', 'enter', 'data', 'copyin', '(', 'a', ')', 'copyout', '(', 'b', '(', '-', '1', ':', ')', ')'],
        ]
        for i,stmt in enumerate(statements):
            #print(util.parsing.tokenize(stmt))
            self.assertEqual(util.parsing.tokenize(stmt),results[i])
    def test_02_next_tokens_till_open_bracket_is_closed(self):
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
    def test_03_split_fortran_line(self):
        for line in self.prepare(testdata1):
            indent,stmt_or_dir,comment,trailing_ws =\
              util.parsing.split_fortran_line(line)
            if "stmt_or_dir" in line:
                self.assertTrue(len(stmt_or_dir))
            else:
                self.assertFalse(len(stmt_or_dir))
            if "comment" in line:
                self.assertTrue(len(comment))
            else:
                self.assertFalse(len(comment))
        # 
    def test_04_detect_line_starts(self):
        input_lines = testdata2.splitlines()
        linenos = util.parsing.detect_line_starts(\
                   input_lines)
        #print(linenos)
        #print([a for i,a in enumerate(input_lines) if i in linenos]) 
        self.assertEqual(linenos,[0,6,9])
    def test_05_relocate_inline_comments(self):
        result = util.parsing.relocate_inline_comments(\
                   testdata2.splitlines())
        self.assertEqual(self.clean("\n".join(result)),self.clean(testdata2_result))
    def test_06_get_top_level_operands(self):
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

    def test_07_extract_function_calls(self):
        for c in ["a","b","c","d","f","g","h","i","j","k"]:
        #for c in ["a"]:
            result = util.parsing.extract_function_calls(testdata3,c)
            #print(result)
            self.assertEqual(result,testdata3_result[c])
    def test_08_parse_use_statement(self):
        statements = [
          "use mymod",
          "use mymod, only: var1",
          "use mymod, only: var1, var2",
          "use mymod, only: var1, var2=>var3",
          "use mymod, var4 => var5",
          "use mymod, only: var4 => var5, operator(+), assignment(=>)",
          "USE module_domain, ONLY : domain, get_ijk_from_grid",
          "use, intrinsic :: iso_c_binding",
          "use, intrinsic :: iso_c_binding, only: myptr_t => c_ptr",
          "use, intrinsic :: iso_c_binding, myptr_t => c_ptr",
        ]
        results = [
          ('mymod', [], [], []),
          ('mymod', [], [], [('var1', 'var1')]),
          ('mymod', [], [], [('var1', 'var1'),('var2', 'var2')]),
          ('mymod', [], [], [('var1', 'var1'),('var2', 'var3')]),
          ('mymod', [], [('var4', 'var5')], []),
          ('mymod', [], [], [('var4', 'var5'), ('operator(+)', 'operator(+)'), ('assignment(=>)', 'assignment(=>)')]),
          ('module_domain', [], [], [('domain', 'domain'), ('get_ijk_from_grid', 'get_ijk_from_grid')]),
          ('iso_c_binding', ['intrinsic'], [], []),
          ('iso_c_binding', ['intrinsic'], [], [('myptr_t', 'c_ptr')]),
          ('iso_c_binding', ['intrinsic'], [('myptr_t', 'c_ptr')], []),
        ]
        for i,stmt in enumerate(statements):
            #print(util.parsing.parse_use_statement(stmt))
            self.assertEqual(util.parsing.parse_use_statement(stmt),results[i])
    def test_09parse_fortran_argument(self):
        statements = [
          "kind(hipSuccess)",
          "kind=4",
          "len=*",
          "kind=2*dp+size(arr,dim=1)",
        ]
        results = [
          (None, 'kind(hipSuccess)'),
          ('kind', '4'),
          ('len', '*'),
          ('kind', '2*dp+size(arr,dim=1)'),
        ]
        for i,stmt in enumerate(statements):
            #print(util.parsing.parse_fortran_argument(stmt))
            self.assertEqual(util.parsing.parse_fortran_argument(stmt),results[i])
    def test_10map_fortran_args_to_positional_args(self):
        expressions = [
          "kind=4",
          "len=*",
          "kind=2*dp+size(arr,dim=1)",
          "kind=a*b,len=5",
          "5,kind=a*b",
          "5,a*b",
          ["5","a*b"],
          [("len","5"),("kind","a*b")],
          [(None,"5"),(None,"a*b")],
          [("kind","a*b"),("len","5")],
        ]
        positional_args = ["len","kind"]
        
        results = [
          [None,'4'],
          ['*',None],
          [None,'2*dp+size(arr,dim=1)'],
          ['5', 'a*b'],
          ['5', 'a*b'],
          ['5', 'a*b'],
          ['5', 'a*b'],
          ['5', 'a*b'],
          ['5', 'a*b'],
          ['5', 'a*b'],
        ]
        for i,expr in enumerate(expressions):
            result = util.parsing.map_fortran_args_to_positional_args(expr,positional_args)
            #print(result)
            self.assertEqual(result,results[i])
        expressions_fail = [
            "len=*,4",    
            "kind=4,*",    
        ] 
        for expr in expressions_fail:
            try:
                util.parsing.map_fortran_args_to_positional_args(expr,positional_args)
                raise Exception("parsing '{}' should have failed".format(expr))
            except:
                pass
    def test_11parse_fortran_character_type(self):
        expressions = [
          # f77
          "character*dp",
          "character*4",
          "character*(*)",
          # modern fortran
          "character(len=*)",
          "CHARACTER(KIND=C_CHAR,LEN=*)",
          "CHARACTER(3,KIND=C_CHAR)",
          "CHARACTER(3,C_CHAR)",
        ]
        results = [
          ('character', 'dp', None,    [], 3),
          ('character', '4', None,     [], 3),
          ('character', '*', None,     [], 5),
          ('character', '*', None,     [], 6),
          ('CHARACTER', '*', 'C_CHAR', [], 10),
          ('CHARACTER', '3', 'C_CHAR', [], 8),
          ('CHARACTER', '3', 'C_CHAR', [], 6),
        ]
        for i,expr in enumerate(expressions):
            #print(util.parsing.parse_fortran_character_type(expr,parse_all=True))
            self.assertEqual(util.parsing.parse_fortran_character_type(expr,parse_all=True),results[i])
        expressions_fail = [
          "character*dp+1",
          "character**",
          "character*:",
          "character*(len=*)",
          "character(kind=C_CHAR,*)",
          "character(len=*,C_CHAR)",
        ] 
        for expr in expressions_fail:
            try:
                util.parsing.parse_fortran_character_type(expr,parse_all=True)
                raise Exception("parsing '{}' should have failed".format(expr))
            except:
                pass
    def test_12parse_fortran_basic_type(self):
        expressions = [
          # f77
          "logical*1",
          "real*1",
          "complex*1",
          "integer*dp",
          "integer*4",
          "integer*(2*2+dp)",
          # modern fortran
          "logical(C_BOOL)",
          "real(C_FLOAT)",
          "complex(c_double_complex)",
          "INTEGER(C_INT)",
          "INTEGER(KIND=C_INT)",
          "double precision",
          "DOUBLE COMPLEX",
        ]
        results = [
          ('logical', None, '1', [], 3),
          ('real', None, '1', [], 3),
          ('complex', None, '1', [], 3),
          ('integer', None, 'dp', [], 3),
          ('integer', None, '4', [], 3),
          ('integer', None, '2*2+dp', [], 9),
          ('logical', None, 'C_BOOL', [], 4),
          ('real', None, 'C_FLOAT', [], 4),
          ('complex', None, 'c_double_complex', [], 4),
          ('INTEGER', None, 'C_INT', [], 4),
          ('INTEGER', None, 'C_INT', [], 6),
          ('real', None, 'c_double', [], 2),
          ('complex', None, 'c_double_complex', [], 2),
        ]
        for i,expr in enumerate(expressions):
            #print(util.parsing.parse_fortran_basic_type(expr,parse_all=True))
            self.assertEqual(util.parsing.parse_fortran_basic_type(expr),results[i])
        expressions_fail = [
          "integer*dp+1",
          "integer**",
          "integer*:",
          "integer*(kind=4)",
          "integer(kind=4",
          "integer(kind=4))",
          "double complex (kind=dp)",
        ] 
        for expr in expressions_fail:
            try:
                util.parsing.parse_fortran_basic_type(expr,parse_all=True)
                raise Exception("parsing '{}' should have failed".format(expr))
            except:
                pass
    def test_13parse_fortran_derived_type(self):
        expressions = [
          "type(mytype)",
          "type(mytype(m))",
          "type(mytype(m,n,k=a*2+c))",
          "TYPE(MYTYPE(M,N,K=A*2+C))",
        ]
        results = [
          ('type', None, 'mytype', [], 4),
          ('type', None, 'mytype', ['m'], 7),
          ('type', None, 'mytype', ['m', 'n', 'k=a*2+c'], 17),
          ('TYPE', None, 'MYTYPE', ['M', 'N', 'K=A*2+C'], 17),
        ]
        for i,expr in enumerate(expressions):
            #print(util.parsing.parse_fortran_derived_type(expr,parse_all=True))
            self.assertEqual(util.parsing.parse_fortran_derived_type(expr),results[i])

        expressions_fail = [
          "type(mytype",
          "type(mytype))",
          "type(kind=mytype(m,n,k))",
          "TYPE(MYTYPE(M,KIND=N))",
        ] 
        for expr in expressions_fail:
            try:
                util.parsing.parse_fortran_derived_type(expr,parse_all=True)
                raise Exception("parsing '{}' should have failed".format(expr))
            except:
                pass
    def test_14_parse_data_type(self):
        expressions = [
          # f77
          "character*dp",
          "character*4",
          "character*(*)",
          # modern fortran
          "character(len=*)",
          "CHARACTER(KIND=C_CHAR,LEN=*)",
          "CHARACTER(3,KIND=C_CHAR)",
          "CHARACTER(3,C_CHAR)",
          #
          # f77
          "logical*1",
          "real*1",
          "complex*1",
          "integer*dp",
          "integer*4",
          "integer*(2*2+dp)",
          # modern fortran
          "logical(C_BOOL)",
          "real(C_FLOAT)",
          "complex(c_double_complex)",
          "INTEGER(C_INT)",
          "INTEGER(KIND=C_INT)",
          "integer(kind(hipSuccess))",
          "double precision",
          "DOUBLE COMPLEX",
          #
          "type(mytype)",
          "type(mytype(m))",
          "type(mytype(m,n,k=a*2+c))",
          "TYPE(MYTYPE(M,N,K=A*2+C))",
        ]
        results = [
          ('character', 'dp', None, [], 3),
          ('character', '4', None, [], 3),
          ('character', '*', None, [], 5),
          ('character', '*', None, [], 6),
          ('CHARACTER', '*', 'C_CHAR', [], 10),
          ('CHARACTER', '3', 'C_CHAR', [], 8),
          ('CHARACTER', '3', 'C_CHAR', [], 6),
          ('logical', None, '1', [], 3),
          ('real', None, '1', [], 3),
          ('complex', None, '1', [], 3),
          ('integer', None, 'dp', [], 3),
          ('integer', None, '4', [], 3),
          ('integer', None, '2*2+dp', [], 9),
          ('logical', None, 'C_BOOL', [], 4),
          ('real', None, 'C_FLOAT', [], 4),
          ('complex', None, 'c_double_complex', [], 4),
          ('INTEGER', None, 'C_INT', [], 4),
          ('INTEGER', None, 'C_INT', [], 6),
          ('integer', None, 'kind(hipSuccess)', [], 7),
          ('real', None, 'c_double', [], 2),
          ('complex', None, 'c_double_complex', [], 2),
          ('type', None, 'mytype', [], 4),
          ('type', None, 'mytype', ['m'], 7),
          ('type', None, 'mytype', ['m', 'n', 'k=a*2+c'], 17),
          ('TYPE', None, 'MYTYPE', ['M', 'N', 'K=A*2+C'], 17),
        ]
        for i,expr in enumerate(expressions):
            #print(util.parsing.parse_fortran_datatype(expr,parse_all=True))
            self.assertEqual(util.parsing.parse_fortran_datatype(expr),results[i])
    def test_15_parse_declaration(self):
        statements = [
          "integer,parameter :: a(1) = (/1/), b = 5*2**3",
          "integer(kind(hipSuccess)),parameter :: ierr = hipSuccess",
          "integer(kind=4),parameter :: mykind = 3",
          "integer*4,pointer :: a(:) => null(), b => null()",
          "integer*4,allocatable :: b(:,:,n,-1:5)",
          "integer,dimension(:,:) :: int_array2d",
          "character*(*) :: a",
          "character(len=*) :: a",
          "integer a( m ), b( m, n )",
          "real*8 a( * ), b( * )",
          "character*(*), intent(in)    :: c",
          "character*(*), c",
          "CHARACTER*4, C",
          "character(len=3,kind=c_char) :: a(3,4*l+3)*k",
          "TYPE(MYTYPE(M,N,K=A*B+C)) :: A(3,4*L+3)",
          "type(mytype),intent(IN out) :: A(3,4)",
          "class(myclass),intent(IN out) :: this",
          "character(len=:),pointer :: a(:,:) => b",
        ]
        results = [
          ('integer', None, None, [], ['parameter'], [], [('a', ['1'], '(/1/)'), ('b', [], '5*2**3')], 'integer', ['parameter']),
          ('integer', None, 'kind(hipSuccess)', [], ['parameter'], [], [('ierr', [], 'hipSuccess')], 'integer(kind(hipSuccess))', ['parameter']),
          ('integer', None, '4', [], ['parameter'], [], [('mykind', [], '3')], 'integer(kind=4)', ['parameter']),
          ('integer', None, '4', [], ['pointer'], [], [('a', [':'], 'null()'), ('b', [], 'null()')], 'integer*4', ['pointer']),
          ('integer', None, '4', [], ['allocatable'], [], [('b', [':', ':', 'n', '-1:5'], None)], 'integer*4', ['allocatable']),
          ('integer', None, None, [], [], [':', ':'], [('int_array2d', [], None)], 'integer', ['dimension(:,:)']),
          ('character', '*', None, [], [], [], [('a', [], None)], 'character*(*)', []),
          ('character', '*', None, [], [], [], [('a', [], None)], 'character(len=*)', []),
          ('integer', None, None, [], [], [], [('a', ['m'], None), ('b', ['m', 'n'], None)], 'integer', []),
          ('real', None, '8', [], [], [], [('a', ['*'], None), ('b', ['*'], None)], 'real*8', []),
          ('character', '*', None, [], ['intent(in)'], [], [('c', [], None)], 'character*(*)', ['intent(in)']),
          ('character', '*', None, [], [], [], [('c', [], None)], 'character*(*)', []),
          ('CHARACTER', '4', None, [], [], [], [('C', [], None)], 'CHARACTER*4', []),
          ('character', 'k', 'c_char', [], [], [], [('a', ['3', '4*l+3'], None)], 'character(len=3,kind=c_char)', []),
          ('TYPE', None, 'MYTYPE', ['M', 'N', 'K=A*B+C'], [], [], [('A', ['3', '4*L+3'], None)], 'TYPE(MYTYPE(M,N,K=A*B+C))', []),
          ('type', None, 'mytype', [], ['intent(INout)'], [], [('A', ['3', '4'], None)], 'type(mytype)', ['intent(IN out)']),
          ('class', None, 'myclass', [], ['intent(INout)'], [], [('this', [], None)], 'class(myclass)', ['intent(IN out)']),
          ('character', ':', None, [], ['pointer'], [], [('a', [':', ':'], 'b')], 'character(len=:)', ['pointer']),
        ]
        for i,stmt in enumerate(statements):
            #print(util.parsing.parse_declaration(stmt))
            self.assertEqual(util.parsing.parse_declaration(stmt),results[i])
    def test_16_parse_do_statement(self):
        statements = [
          "do",
          "DO",
          "DO jj = MAX(jts,1) , MIN(jte,jde-1,spec_bdy_width)",
          "label: do j = min(x,y,z,k), max(M,n), min(a0,a1,2)",
          "DO 2000 IIM=1,NF",
        ]
        results = [
          (None, None, None, None, None),
          (None, None, None, None, None),
          (None, 'jj', 'MAX(jts,1)', 'MIN(jte,jde-1,spec_bdy_width)', None),
          ('label', 'j', 'min(x,y,z,k)', 'max(M,n)', 'min(a0,a1,2)'),
          ('2000', 'IIM', '1', 'NF', None),
        ]
        for i,stmt in enumerate(statements):
            #print(util.parsing.parse_do_statement(stmt))
            self.assertEqual(util.parsing.parse_do_statement(stmt),results[i])
    def test_17_parse_type_statement(self):
        statements = [
          "type a",
          "type :: a",
          "type,bind(c) :: a",
          "type,bind(c) :: a(k,l)",
          "type,bind(c),private :: a(k,l)",
        ]
        results = [
          ('a', [], []),
          ('a', [], []),
          ('a', ['bind(c)'], []),
          ('a', ['bind(c)'], ['k', 'l']),
          ('a', ['bind(c)', 'private'], ['k', 'l']),
        ]
        for i,stmt in enumerate(statements):
            #print(util.parsing.parse_type_statement(stmt))
            self.assertEqual(util.parsing.parse_type_statement(stmt),results[i])
    def test_18_parse_attributes_statement(self):
        statements = [
          "attributes(device,constant) :: a_d, b_d"
        ]
        results = [
          (['device','constant'], ['a_d','b_d'])
        ]
        for i,stmt in enumerate(statements):
            #print(util.parsing.parse_attributes_statement(stmt))
            self.assertEqual(util.parsing.parse_attributes_statement(stmt),results[i])
    def test_19_parse_function_statement(self):
        statements = [
          "function foo",
          "function foo()",
          "logical function foo ( arg1, arg2, arg3 )",
          "attributes(device) function foo()",
          "function foo( arg1, arg2, arg3 ) result(i5) bind(c,name=\"cname\")",
          "integer * 5 pure recursive function foo( arg1, arg2, arg3 ) result(i5) bind(c,name=\"cname\")",
          "integer(kind(hipSuccess)) attributes(host,device) pure recursive function foo( arg1, arg2, arg3 ) result(i5) bind(c,name=\"cname\")",
          "attributes(host,device) pure recursive subroutine foo( arg1, arg2, arg3 ) bind(c,name=\"cname\")",
        ]
        results = [
            ('function', 'foo', [], [], [], (None, None, 'foo'), (False, None)),
            ('function', 'foo', [], [], [], (None, None, 'foo'), (False, None)),
            ('function', 'foo', ['arg1', 'arg2', 'arg3'], [], [], ('logical', None, 'foo'), (False, None)),
            ('function', 'foo', [], [], ['device'], (None, None, 'foo'), (False, None)),
            ('function', 'foo', ['arg1', 'arg2', 'arg3'], [], [], (None, None, 'i5'), (True, 'cname')),
            ('function', 'foo', ['arg1', 'arg2', 'arg3'], ['pure', 'recursive'], [], ('integer', '5', 'i5'), (True, 'cname')),
            ('function', 'foo', ['arg1', 'arg2', 'arg3'], ['pure', 'recursive'], ['host', 'device'], ('integer', 'kind(hipSuccess)', 'i5'), (True, 'cname')),
            ('subroutine', 'foo', ['arg1', 'arg2', 'arg3'], ['pure', 'recursive'], ['host', 'device'], (None, None, None), (True, 'cname')),
        ]
        for i,stmt in enumerate(statements):
           #print(util.parsing.parse_function_statement(stmt))
           self.assertEqual(util.parsing.parse_function_statement(stmt),results[i])
    def test_20_strip_array_indexing(self):
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
    def test_21_derived_type_parents(self):
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
    def test_22_parse_directive(self):
        expressions = [
          "!$acc enter data copyin(a,b,c(:)) copyout(b(-1:))",
          "!$ACC ENTER data copyin(a,b,c(:)) copyout(b(-1:))",
          "!$ACC ENTER dAta copyin(a,b,c(:)), copyout(b(-1:))",
        ]
        results = [
          ['!$', 'acc', 'enter', 'data', 'copyin(a,b,c(:))', 'copyout(b(-1:))'],
          ['!$', 'ACC', 'ENTER', 'data', 'copyin(a,b,c(:))', 'copyout(b(-1:))'],
          ['!$', 'ACC', 'ENTER', 'dAta', 'copyin(a,b,c(:))', 'copyout(b(-1:))'],
        ]
        for i,expr in enumerate(expressions):
            #print(util.parsing.parse_directive(expr))
            self.assertEqual(util.parsing.parse_directive(expr),results[i])
    
    def test_23_parse_acc_clauses(self):
        expressions = [
          ["copyin(a,b,c(:))","copyout(b(-1:))","async"],
          ["copyin(a,b,c(:))","copyout(b(-1:))","reduction(+:a)","async"],
          ["map(to:x,y(:),tofrom:a%z(1:n,2:m))"], # actually OMP clauses
        ]
        results = [
          [('copyin', ['a', 'b', 'c(:)']), ('copyout', ['b(-1:)']), ('async', [])],
          [('copyin', ['a', 'b', 'c(:)']), ('copyout', ['b(-1:)']), ('reduction', [('+', ['a'])]), ('async', [])],
          [('map', [('to', ['x', 'y(:)']), ('tofrom', ['a%z(1:n,2:m)'])])],
        ]
        for i,expr in enumerate(expressions):
            #print(util.parsing.parse_acc_clauses(expr))
            self.assertEqual(util.parsing.parse_acc_clauses(expr),results[i])
    
    def test_24_parse_acc_directive(self):
        expressions = [
          "!$acc enter data copyin(a,b,c(:)) copyout(b(-1:))",
          "!$acc wait(i,j) async(c)",
          "!$acc kernels loop reduction(+:x)",
          "!$acc parallel loop collapse(3) gang vector copyin(grid%al,grid%alb,grid%pb,grid%t_1) copyin(moist(:,:,:,p_qv))"
        ]
        results = [
          ('!$', ['acc', 'enter', 'data'], [], ['copyin(a,b,c(:))', 'copyout(b(-1:))']),
          ('!$', ['acc', 'wait'], ['i', 'j'], ['async(c)']),
          ('!$', ['acc', 'kernels', 'loop'], [], ['reduction(+:x)']),
          ('!$', ['acc', 'parallel', 'loop'], [], ['collapse(3)', 'gang', 'vector', 'copyin(grid%al,grid%alb,grid%pb,grid%t_1)', 'copyin(moist(:,:,:,p_qv))']),
        ]
        for i,expr in enumerate(expressions):
            #print(util.parsing.parse_acc_directive(expr))
            self.assertEqual(util.parsing.parse_acc_directive(expr),results[i])
    def test_25_parse_cuf_kernel_call(self):
        expressions = [
          "call mykernel<<<grid,block>>>(arg1,arg2,arg3(1:n))",
          "call mykernel<<<grid,block,0,stream>>>(arg1,arg2,arg3(1:n))",
        ]
        results = [
          ('mykernel',['grid','block'],['arg1','arg2','arg3(1:n)']),
          ('mykernel',['grid','block','0','stream'],['arg1','arg2','arg3(1:n)']),
        ]
        for i,expr in enumerate(expressions):
            #print(util.parsing.parse_cuf_kernel_call(expr))
            self.assertEqual(util.parsing.parse_cuf_kernel_call(expr),results[i])
    def test_26_mangle_fortran_var_expr(self):
        expressions = [
          "a(i,j)%b%arg3(1:n)",
        ]
        results = [
          "aLijR_b_arg3L1TnR",
        ]
        for i,expr in enumerate(expressions):
            #print(util.parsing.mangle_fortran_var_expr(expr))
            self.assertEqual(util.parsing.mangle_fortran_var_expr(expr),results[i])
    def test_27parse_fortran_derived_type_statement(self):
        expressions = [
          'type mytype',
          'type :: mytype',
          'type, bind(c) :: mytype',
          'type :: mytype(k,l)',
        ]
        results = [
          ('mytype', [], []),
          ('mytype', [], []),
          ('mytype', ['bind(c)'], []),
          ('mytype', [], ['k','l']),
        ]
        for i,expr in enumerate(expressions):
            #print(util.parsing.parse_derived_type_statement(expr))
            self.assertEqual(util.parsing.parse_derived_type_statement(expr),results[i])
    
    def test_28_parse_allocate_statement(self):
        expressions = [
          'allocate(a(1:N))',
          'allocate(a(1:N),b(-1:m:2,n))',
          'allocate(a(1:N),b(-1:m:2,n),stat=ierr)',
          'allocate(grid(domain%num,j)%a(1:N))',
          'allocate ( new_grid )',
          'allocate( new_grid%cell )',
        ]
        results = [
          ([('a', [('1', 'N', None)])], None),
          ([('a', [('1', 'N', None)]), ('b', [('-1', 'm', '2'), (None, 'n', None)])], None),
          ([('a', [('1', 'N', None)]), ('b', [('-1', 'm', '2'), (None, 'n', None)])], 'ierr'),
          ([('grid(domain%num,j)%a', [('1', 'N', None)])], None),
          ([('new_grid', [])], None),
          ([('new_grid%cell', [])], None),
        ]
        for i,expr in enumerate(expressions):
            #print(util.parsing.parse_allocate_statement(expr))
            self.assertEqual(util.parsing.parse_allocate_statement(expr),results[i])
    def test_29_parse_deallocate_statement(self):
        expressions = [
          'deallocate(a)',
          'deallocate(a,b)',
          'deallocate(a,b,stat=ierr)',
          'deallocate(grid(domain%num,j)%a)',
        ]
        results = [
          (['a'], None),
          (['a', 'b'], None),
          (['a', 'b'], 'ierr'),
          (['grid(domain%num,j)%a'], None),
        ]
        for i,expr in enumerate(expressions):
            #print(util.parsing.parse_deallocate_statement(expr))
            self.assertEqual(util.parsing.parse_deallocate_statement(expr),results[i])
    def test_30_parse_lvalue(self):
        expressions = [
          "a",
          "a(i,j)",
          "a(i,j)%b",
          "a(i,j)%b(m,n,k)",
          "a(i,j)%b(2*m+k,n**3,k**4+1+6*k)",
        ]
        results = [
          [('a', [])],
          [('a', ['i', 'j'])],
          [('a', ['i', 'j']), ('b', [])],
          [('a', ['i', 'j']), ('b', ['m', 'n', 'k'])],
          [('a', ['i', 'j']), ('b', ['2*m+k', 'n**3', 'k**4+1+6*k'])],
        ]
        for i,expr in enumerate(expressions):
            #print(util.parsing.parse_lvalue(expr))
            self.assertEqual(util.parsing.parse_lvalue(expr),results[i])
        # expressions that should fail
        expressions_fail = [
          "a b",
          "where(a(i,j)>0) a",
          "do i",
        ]
        for expr in expressions_fail:
            try:
                util.parsing.parse_lvalue(expr)
                raise Exception("parsing '{}' should have failed".format(expr))
            except:
                pass
    def test_31_is_assignment(self):
        expressions = [
          "a = 1",
          "a(i,j) = 1",
          "a(i,j)%b = 1",
          "a(i,j)%b(m,n,k) = 1",
          "a(i,j)%b(2*m+k,n**3,k**4+1+6*k) = 1",
          "a b = 1",
          "do i = 1,2",
          "REAL real = 5",
          "parameter c = 5.0",
        ]
        results = [
          True,
          True,
          True,
          True,
          True,
          False,
          False,
          False,
          False,
        ]
        for i,expr in enumerate(expressions):
            self.assertEqual(util.parsing.is_assignment(expr),results[i])

    def test_32_parse_parameter_statement(self):
        expressions = [
          "PARAMETER( a = 5, b = 3)",
          "PARAMETER a = 5, b = 3", # legacy version
        ]
        results = [
          [('a','5'), ('b','3')],
          [('a','5'), ('b','3')],
        ]
        for i,expr in enumerate(expressions):
            #print(util.parsing.parse_parameter_statement(expr))
            self.assertEqual(util.parsing.parse_parameter_statement(expr),results[i])
    def test_33_parse_public_statement(self):
        expressions = [
          "public",
          "PUBLIC a",
          "public a,b",
          "PUBLIC :: a",
          "public :: a,b",
          "public :: operator(.eq.)",
          "public operator(.eq.), a, operator(-)",
          "public operator(.eq.), a, operator(-), assignment(=), assignment(=>)",
        ]
        results = [
          ("public",[]),
          ("PUBLIC",["a"]),
          ("public",["a","b"]),
          ("PUBLIC",["a"]),
          ("public",["a","b"]),
          ('public', ['operator(.eq.)']),
          ('public', ['operator(.eq.)','a','operator(-)']),
          ('public', ['operator(.eq.)','a','operator(-)','assignment(=)','assignment(=>)'])
        ]
        for i,expr in enumerate(expressions):
            result = util.parsing.parse_public_statement(expr)
            #print(result)
            self.assertEqual(result,results[i])
    def test_34_expand_letter_range(self):
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
    def test_35_parse_implicit_statement(self):
        expressions = [
          "implicit none",
          "IMPLICIT NONE",
          "IMPLICIT integer (i,m-p)",
          "IMPLICIT double precision (a-h,o-z)",
          "IMPLICIT DOUBLE COmplex (a-h,o-z)",
          "IMPLICIT DOUBLE COmplex (a-h,o-z), character (i)",
          "IMPLICIT integer (i-n), real (a-h,o-z)",
          "IMPLICIT integer(kind=8) (i,m-p,a)",
          "IMPLICIT integer(kind=8) (i,m-p,a), character (c), type(mytype) (d)",
        ]
        results = [
          [(None, None, None, [])],
          [(None, None, None, [])],
          [('integer', None, None, ['i', 'm', 'n', 'o', 'p'])],
          [('real', None, 'c_double', ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])],
          [('complex', None, 'c_double_complex', ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])],
          [('complex', None, 'c_double_complex', ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']), ('character', None, None, ['i'])],
          [('integer', None, None, ['i', 'j', 'k', 'l', 'm', 'n']), ('real', None, None, ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])],
          [('integer', None, '8', ['i', 'm', 'n', 'o', 'p', 'a'])],
          [('integer', None, '8', ['i', 'm', 'n', 'o', 'p', 'a']), ('character', None, None, ['c']), ('type', None, 'mytype', ['d'])],
        ]
        for i,expr in enumerate(expressions):
            #print(util.parsing.parse_implicit_statement(expr))
            self.assertEqual(util.parsing.parse_implicit_statement(expr),results[i])
    def test_36_parse_interface_statement(self):
        expressions = [
          "interface",
          "INTERFACE myInterFace",
        ]
        results = [
          None,
          "myInterFace",
        ]
        for i,expr in enumerate(expressions):
            #print(util.parsing.parse_interface_statement(expr))
            self.assertEqual(util.parsing.parse_interface_statement(expr),results[i])
    def test_37_parse_case_statement(self):
        expressions = [
          "case (0)",
          "case (0,1,2)",
          "case (0,-1,2,A,B,C,D)",
        ]
        results = [
          ['0'],
          ['0', '1', '2'],
          ['0', '-1', '2', 'A', 'B', 'C', 'D'],
        ]
        for i,expr in enumerate(expressions):
            #print(util.parsing.parse_case_statement(expr))
            self.assertEqual(util.parsing.parse_case_statement(expr),results[i])
    def test_38_parse_dimension_statement(self):
        expressions = [
          "dim a(1)",
          "DIMENSION a(2,3)",
          "DIMENSION a(2,3), b(c,d,e:h)",
        ]
        results = [
          [('a', ['1'])],
          [('a', ['2', '3'])],
          [('a', ['2', '3']), ('b', ['c', 'd', 'e:h'])],
        ]
        for i,expr in enumerate(expressions):
            #print(util.parsing.parse_dimension_statement(expr))
            self.assertEqual(util.parsing.parse_dimension_statement(expr),results[i])
    def test_38_parse_external_statement(self):
        expressions = [
          "EXTERNAL a",
          "external :: a",
          "external a, b, c",
          "EXTERNAL :: a, b, c",
        ]
        results = [
          ['a'],
          ['a'],
          ['a', 'b', 'c'],
          ['a', 'b', 'c'],
        ]
        for i,expr in enumerate(expressions):
            #print(util.parsing.parse_external_statement(expr))
            self.assertEqual(util.parsing.parse_external_statement(expr),results[i])
    
if __name__ == '__main__':
    unittest.main() 
