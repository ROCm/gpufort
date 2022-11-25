# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from .base import *
from .arithexpr import *
from .fortran import *
from .acc import *
from .cuf import *
from .directives import *
from . import traversals

from gpufort import grammar as _grammar

grammar = _grammar.Grammar(
  ignorecase=True,
  unary_op_parse_action=TTUnaryOp,   
  binary_op_parse_action=TTBinaryOpChain
)
grammar_no_logic = _grammar.Grammar(
  ignorecase=True,
  unary_op_parse_action=TTUnaryOp,   
  binary_op_parse_action=TTBinaryOpChain,
  no_logic_ops=True
)
grammar_no_custom = _grammar.Grammar(
  ignorecase=True,
  unary_op_parse_action=TTUnaryOp,   
  binary_op_parse_action=TTBinaryOpChain,
  no_custom_ops=True
)
grammar_no_logic_no_custom = _grammar.Grammar(
  ignorecase=True,
  unary_op_parse_action=TTUnaryOp,   
  binary_op_parse_action=TTBinaryOpChain,
  no_logic_ops=True,
  no_custom_ops=True
)
del _grammar    

set_arith_expr_parse_actions(grammar)
set_acc_parse_actions(grammar)
set_cuf_parse_actions(grammar)

set_arith_expr_parse_actions(grammar_no_logic)
set_acc_parse_actions(grammar_no_logic)
set_cuf_parse_actions(grammar_no_logic)

set_arith_expr_parse_actions(grammar_no_custom)
set_acc_parse_actions(grammar_no_custom)
set_cuf_parse_actions(grammar_no_custom)

set_arith_expr_parse_actions(grammar_no_logic_no_custom)
set_acc_parse_actions(grammar_no_logic_no_custom)
set_cuf_parse_actions(grammar_no_logic_no_custom)
