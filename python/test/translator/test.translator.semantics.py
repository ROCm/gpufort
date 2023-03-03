#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os,sys
import time
import unittest
import addtoplevelpath
import inspect
import cProfile,pstats,io

from gpufort import translator
from gpufort import indexer
from gpufort import util

print("Running test '{}'".format(os.path.basename(__file__)),end="",file=sys.stderr)

PROFILING_ENABLE = False

scope = None

semantics = translator.semantics.Semantics()

class TestSemantics(unittest.TestCase):
    
    def setUp(self):
        global PROFILING_ENABLE
        self.started_at = time.time()
        if PROFILING_ENABLE:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
    
    def tearDown(self):
        elapsed = time.time() - self.started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 9)))
        if PROFILING_ENABLE:
            self.profiler.disable() 
            s = io.StringIO()
            sortby = 'cumulative'
            stats = pstats.Stats(self.profiler, stream=s).sort_stats(sortby)
            stats.print_stats(10)
            print(s.getvalue())
    
    def test_00_do_nothing(self):
        pass
    
    def test_01_create_scope(self):
        global scope
        scope = indexer.scope.create_scope_from_declaration_list("""\
        use iso_c_binding
        implicit none
        integer, parameter :: m = 2, n = 3, p = 4
        !integer, parameter :: Q(3) = [1,2,3]
        integer :: i,j,k
        integer :: A(m),B(m,n),C(m,n,k)
        integer(c_long) :: D(m),E(m,n),F(m,n,p)
        real :: alpha,beta,gamma
        real :: X(m),Y(m,n),Z(m,n,p)
        real(c_double) :: U(m),V(m,n),W(m,n,p)

        type mytype_t
          real :: tX(m),tY(m,n),tZ(m,n,p)
        end type
        
        type(mytype_t) :: mytypes(10)
        """)
    
    parse_arith_expr_testdata = [
      "1", 
      "(1.0,2.0)", # all literals
      "(1.0_c_double,2.0)",
      "1 + 2  * 3",
      "1.0 + 2.0 * 3.0",
      "[1.0,2.0,3.0]", # till here
      "[1.0,2.0,1 + 2.0 + 3.0]", # till here
      "[1,2,M,N,P]",  # all parameters
      "M + N*P", 
      "[M + N*P,4*N*P]", # till here
      "[1,2,i,j,k]", 
      "i + j*k",
      "[i + j*k,4*j*k]",
      "A(i) + B(i,j)",
      "A(i) + B(i,j) - C(i,j,k)",
      "A(i) + B(i,j)*C(i,j,k)",
      "A(i) + B(i,j)*C(i,j,k) + A(k)",
      "C(i,j,k)*(A(i) + B(i,j)*C(i,j,k) + A(k))",
      # rank > 0
      "X(:) + alpha",
      "A(:) + B(1,:)",
      "X(:) + Y(1,:) + Z(1,n,:)",
      "Y + Z(1,:,:)",
      "V + W(1,:,:)",
      "V + Z(1,:,:)", # mix float and double
    ]       
    
    parse_arith_expr_parse_results = [] # will be filled by a test
    
    #type|bytes per element|rank|parameter|literal|unprefixed_single_val
    resolve_arith_expr_results = [
      ('integer', 4, 0, True, True, True, 1),
      ('complex', 8, 0, True, True, True, (1+2j)),
      ('complex', 16, 0, True, True, True, (1+2j)),
      ('integer', 4, 0, True, True, False, 7),
      ('real', 4, 0, True, True, False, 7.0),
      ('real', 4, 1, True, True, True, [1.0, 2.0, 3.0]),
      ('real', 4, 1, True, True, True, [1.0, 2.0, 6.0]),
      ('integer', 4, 1, True, False, True, None),
      ('integer', 4, 0, True, False, False, None),
      ('integer', 4, 1, True, False, True, None),
      ('integer', 4, 1, False, False, True, None),
      ('integer', 4, 0, False, False, False, None),
      ('integer', 4, 1, False, False, True, None),
      ('integer', 4, 0, False, False, False, None),
      ('integer', 4, 0, False, False, False, None),
      ('integer', 4, 0, False, False, False, None),
      ('integer', 4, 0, False, False, False, None),
      ('integer', 4, 0, False, False, False, None),
      ('real', 4, 1, False, False, False, None),
      ('integer', 4, 1, False, False, False, None),
      ('real', 4, 1, False, False, False, None),
      ('real', 4, 2, False, False, False, None),
      ('real', 8, 2, False, False, False, None),
      ('real', 8, 2, False, False, False, None),
    ]

    def test_02_parse_arith_expr(self):
        for i,test in enumerate(TestSemantics.parse_arith_expr_testdata):
            ttarithexpr = translator.tree.parse_arith_expr(test)
            TestSemantics.parse_arith_expr_parse_results.append(ttarithexpr)
    
    def test_03_resolve_arith_expr(self):
        for i,test in enumerate(TestSemantics.parse_arith_expr_testdata):
            #print(test)
            ttarithexpr = TestSemantics.parse_arith_expr_parse_results[i]
            #print(ttarithexpr.fstr())
            semantics.resolve_arith_expr(ttarithexpr,scope)
            try:
                eval_result = ttarithexpr.eval()
            except Exception as e:
                eval_result = None
            result_tuple = (
              ttarithexpr.type,
              ttarithexpr.bytes_per_element,
              ttarithexpr.rank,
              ttarithexpr.yields_parameter,
              ttarithexpr.yields_literal,
              ttarithexpr.is_unprefixed_single_value,
              eval_result,
            )
            #print(str(result_tuple)+",")
            self.assertEqual(TestSemantics.resolve_arith_expr_results[i],result_tuple)

    parse_rvalue_testdata = [
      # 1d
      "X",
      "X(1)",
      "X(:)",
      "X(1:)",
      "X(1:5)",
      "X(1:5:2)",
      # 2d
      "Y",
      "Y(1,1)",
      "Y(:,:)",
      "Y(:,1)",
      "Y(1:,1)",
      "Y(1:5,1)",
      "Y(:,:5)",
      "Y(1,:5)",
      "Y(1:,:5)", # not contiguous
      "Y(1:5,:5)",
      "Y(:,1:5)", # contiguous
      # derived_type expressions 
      "mytypes(i)%tx(1)",
      "mytypes(i)%tx(:)",
      "mytypes(:)%tx(1)",
    ]
    
    parse_rvalue_parse_results = [] # will be filled by a test
    
    # scalar|array|contiguous array|full array
    resolve_rvalue_results = [
      (False, True, True, True),
      (True, False, False, False),
      (False, True, True, True),
      (False, True, True, False),
      (False, True, True, False),
      (False, True, False, False),
      (False, True, True, True),
      (True, False, False, False),
      (False, True, True, True),
      (False, True, True, False),
      (False, True, True, False),
      (False, True, True, False),
      (False, True, True, False),
      (False, True, False, False),
      (False, True, False, False),
      (False, True, False, False),
      (False, True, True, False),
      (True, False, False, False),
      (False, True, True, True),
      (False, True, False, False),
    ]
    
    def test_04_parse_rvalue(self):
        for i,test in enumerate(TestSemantics.parse_rvalue_testdata):
            ttrvalue = translator.tree.parse_rvalue(test)
            TestSemantics.parse_rvalue_parse_results.append(ttrvalue)

    def test_05_resolve_rvalue(self):
        #print("# scalar|array|contiguous array|full array")
        for i,test in enumerate(TestSemantics.parse_rvalue_testdata):
            ttrvalue = TestSemantics.parse_rvalue_parse_results[i] 
            semantics.resolve_arith_expr(ttrvalue,scope)
            result_tuple = (
              ttrvalue.yields_scalar,
              ttrvalue.yields_array,
              ttrvalue.yields_contiguous_array,
              ttrvalue.yields_full_array,
            )
            #print(test)
            #print(result_tuple)
            self.assertEqual(
              TestSemantics.resolve_rvalue_results[i],
              result_tuple
            )
  
    parse_function_call_testdata = [
      "real(1)",
      "real(1,8)",
      "real(j,kind=8)",
      "real(1,c_double)",
      "real(j,kind=c_double)",
      "real(X,kind=c_double)",
      "real(Y(:,1),kind=c_double)",
      "real(X+Y(:,1),kind=c_double)",
      "real(Y(1,:),kind=c_double)",
      "int(1)",
      "int(1,8)",
      "int(j,kind=8)",
      "int(1,c_long)",
      "int(j,kind=c_long)",
      "int(X,kind=c_long)",
      "int(X+Y(:,1),kind=c_long)",
      "int(Y(:,1),kind=c_long)",
      "int(Y(1,:),kind=c_long)",
      "cmplx(1,2)",
      "cmplx(1,2d0)",
      "cmplx(1,2d0,kind=c_float_complex)",
      "cmplx(Y,kind=c_float_complex)",
      "cmplx(Y(:,1),kind=c_float_complex)",
      "cmplx(Y(1,:),kind=c_float_complex)",
      "cmplx(1,Y,kind=c_float_complex)",
      "cmplx(1,Y(:,1),kind=c_float_complex)",
      "cmplx(1,Y(1,:),kind=c_float_complex)",
    ]
   
    parse_function_call_parse_results = [
    ]
    
    #function|elemental|converter|type|kind|bytes_per_element|rank|is_contiguous|yields_full_array
    resolve_function_call_results = [
      (True, True, True, 'real', None, 4, 0, False, False),
      (True, True, True, 'real', '8', 8, 0, False, False),
      (True, True, True, 'real', '8', 8, 0, False, False),
      (True, True, True, 'real', 'c_double', 8, 0, False, False),
      (True, True, True, 'real', 'c_double', 8, 0, False, False),
      (True, True, True, 'real', 'c_double', 8, 1, True, True),
      (True, True, True, 'real', 'c_double', 8, 1, True, False),
      (True, True, True, 'real', 'c_double', 8, 1, True, False),
      (True, True, True, 'real', 'c_double', 8, 1, False, False),
      (True, True, True, 'integer', None, 4, 0, False, False),
      (True, True, True, 'integer', '8', 8, 0, False, False),
      (True, True, True, 'integer', '8', 8, 0, False, False),
      (True, True, True, 'integer', 'c_long', 8, 0, False, False),
      (True, True, True, 'integer', 'c_long', 8, 0, False, False),
      (True, True, True, 'integer', 'c_long', 8, 1, True, True),
      (True, True, True, 'integer', 'c_long', 8, 1, True, False),
      (True, True, True, 'integer', 'c_long', 8, 1, True, False),
      (True, True, True, 'integer', 'c_long', 8, 1, False, False),
      (True, True, True, 'complex', None, 8, 0, False, False),
      (True, True, True, 'complex', None, 8, 0, False, False),
      (True, True, True, 'complex', 'c_float_complex', 8, 0, False, False),
      (True, True, True, 'complex', 'c_float_complex', 8, 2, True, True),
      (True, True, True, 'complex', 'c_float_complex', 8, 1, True, False),
      (True, True, True, 'complex', 'c_float_complex', 8, 1, False, False),
      (True, True, True, 'complex', 'c_float_complex', 8, 2, True, True),
      (True, True, True, 'complex', 'c_float_complex', 8, 1, True, False),
      (True, True, True, 'complex', 'c_float_complex', 8, 1, False, False),
    ]
 
    def test_06_parse_function_call(self):
        for i,test in enumerate(TestSemantics.parse_function_call_testdata):
            #print(test)
            ttrvalue = translator.tree.parse_rvalue(test)
            TestSemantics.parse_function_call_parse_results.append(ttrvalue)
    
    def test_07_resolve_function_call(self):
        #print("# scalar|array|contiguous array|full array")
        for i,test in enumerate(TestSemantics.parse_function_call_testdata):
            ttrvalue = TestSemantics.parse_function_call_parse_results[i] 
            #print(test)
            semantics.resolve_arith_expr(ttrvalue,scope)
            result_tuple = (
              ttrvalue.is_function_call,
              ttrvalue.is_elemental_function_call,
              ttrvalue.is_converter_call,
              ttrvalue.type,
              ttrvalue.kind,
              ttrvalue.bytes_per_element,
              ttrvalue.rank,
              ttrvalue.yields_contiguous_array,
              ttrvalue.yields_full_array, 
            )
            #print(result_tuple)
            self.assertEqual(
              TestSemantics.resolve_function_call_results[i],
              result_tuple
            )
    
    acc_directive_testdata = [
      "!$acc kernels loop",
      "!$acc kernels loop gang(10) worker vector",
      "!$acc kernels loop gang(10) worker vector_length(32)",
      "!$acc kernels loop gang(10) worker vector_length(32) reduction(min:k)",
    ]

    acc_directive_testdata_wrong_syntax = [
      "!$acc kernels loop gang gang",
      "!$acc kernels loop device_type(radeon) vector(1) vector(2)",
    ]
    
    acc_directive_testdata_wrong_semantics = [
      "!$acc kernels loop copy(a) copyout(a)",
    ]
    
    acc_directive_parse_results = []
    acc_directive_parse_results_wrong_semantics = []
    
    def _acc_assert_fails_to_parse(self,test):
        passed = True
        try:
            translator.tree.parse_acc_directive(test)
        except util.error.SyntaxError as e: 
            passed = False
            pass
        self.assertFalse(passed)
    
    def _acc_assert_fails_to_resolve(self,ttaccdir):
        passed = True
        try:
            semantics.resolve_acc_directive(ttaccdir,scope)
        except util.error.SemanticError as e: 
            #print(e)
            passed = False
            pass
        self.assertFalse(passed)
    
    def test_08_parse_acc_directive(self):
        for i,test in enumerate(TestSemantics.acc_directive_testdata_wrong_syntax):
            self._acc_assert_fails_to_parse(test)
        for i,test in enumerate(TestSemantics.acc_directive_testdata):
            #print(test)
            ttaccdir = translator.tree.parse_acc_directive(test)
            TestSemantics.acc_directive_parse_results.append(ttaccdir)
        for i,test in enumerate(TestSemantics.acc_directive_testdata_wrong_semantics):
            #print(test)
            ttaccdir = translator.tree.parse_acc_directive(test)
            TestSemantics.acc_directive_parse_results_wrong_semantics.append(ttaccdir)
    
    def test_09_resolve_acc_directive(self):
        for i,ttaccdir in enumerate(TestSemantics.acc_directive_parse_results_wrong_semantics):
            self._acc_assert_fails_to_resolve(ttaccdir)
        for i,ttaccdir in enumerate(TestSemantics.acc_directive_parse_results):
            #print(ttaccdir)
            semantics.resolve_acc_directive(ttaccdir,scope)

if __name__ == '__main__':
    unittest.main() 
