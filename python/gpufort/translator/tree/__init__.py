# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import re

from .base import *
from .arithexpr import *
from .transformations import *
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

_p_logic_op = re.compile(r"<=?|=?>|[/=]=|\.(eq|ne|not|and|or|xor|eqv|neqv|[gl][te])\.|\.not\.",re.IGNORECASE)
_p_custom_op = re.compile(r"\.[a-zA-Z]+\.") # may detect logic ops too

def _contains_logic_ops(tokens):
    for tk in tokens:
        if _p_logic_op.match(tk):
            return True 
    return False

def _contains_custom_ops(tokens):
    for tk in tokens:
        if _p_custom_op.match(tk):
            if not _p_logic_op.match(tk):
                return True 
    return False

def parse_extent(expr_as_str,parse_all=True):
    """Parse an extent expression."""
    tokens = util.parsing.tokenize(expr_as_str)
    contains_logic_ops = _contains_logic_ops(tokens)
    contains_custom_ops = _contains_custom_ops(tokens)
    if not contains_custom_ops:
        return grammar_no_logic_no_custom.extent.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    else:
        return grammar_no_logic.extent.parseString(
          expr_as_str, parseAll=parse_all
        )[0]

def parse_arith_expr(expr_as_str,parse_all=True):
    """Parse an arithmetic expression, choose the fastest pyparsing
    parser based on the operators found in the token stream.
    :param expr_as_str: arithmetic expression expression as string.
    :return: The root node of the parse tree.
    """
    tokens = util.parsing.tokenize(expr_as_str)
    contains_logic_ops = _contains_logic_ops(tokens)
    contains_custom_ops = _contains_custom_ops(tokens)
    if not contains_custom_ops and not contains_logic_ops:
        return grammar_no_logic_no_custom.arith_expr.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    elif not contains_logic_ops:
        return grammar_no_logic.arith_expr.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    elif not contains_custom_ops:
        return grammar_no_custom.arith_expr.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    else:
        return grammar.arith_expr.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    
def parse_assignment(expr_as_str,parse_all=True):
    """Parse an assignment, choose the fastest pyparsing
    parser based on the operators found in the token stream.
    :param expr_as_str: assignment expression as string.
    :return: The root node of the parse tree.
    """
    tokens = util.parsing.tokenize(expr_as_str)
    contains_logic_ops = _contains_logic_ops(tokens)
    contains_custom_ops = _contains_custom_ops(tokens)
    if not contains_custom_ops and not contains_logic_ops:
        return grammar_no_logic_no_custom.assignment.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    elif not contains_logic_ops:
        return grammar_no_logic.assignment.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    elif not contains_custom_ops:
        return grammar_no_custom.assignment.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    else:
        return grammar.assignment.parseString(
          expr_as_str, parseAll=parse_all
        )[0]

def parse_rvalue(expr_as_str,parse_all=True):
    """Parse an rvalue, choose the fastest pyparsing
    parser based on the operators found in the token stream.
    :param expr_as_str: rvalue expression as string.
    :return: The root node of the parse tree.
    """
    #:todo: rename
    tokens = util.parsing.tokenize(expr_as_str)
    contains_logic_ops = _contains_logic_ops(tokens)
    contains_custom_ops = _contains_custom_ops(tokens)
    if not contains_custom_ops and not contains_logic_ops:
        return grammar_no_logic_no_custom.rvalue.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    elif not contains_logic_ops:
        return grammar_no_logic.rvalue.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    elif not contains_custom_ops:
        return grammar_no_custom.rvalue.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    else:
        return grammar.rvalue.parseString(
          expr_as_str, parseAll=parse_all
        )[0]

def parse_acc_directive(expr_as_str,parse_all=True):
    """Parse an OpenACC directive, choose the fastest pyparsing
    parser based on the operators found in the token stream.
    :param expr_as_str: OpenACC directive expression as string.
    :return: The root node of the parse tree.
    """
    tokens = util.parsing.tokenize(expr_as_str)
    contains_logic_ops = _contains_logic_ops(tokens)
    contains_custom_ops = _contains_custom_ops(tokens)
    if not contains_custom_ops and not contains_logic_ops:
        return grammar_no_logic_no_custom.acc_directive.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    elif not contains_logic_ops:
        return grammar_no_logic.acc_directive.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    elif not contains_custom_ops:
        return grammar_no_custom.acc_directive.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    else:
        return grammar.acc_directive.parseString(
          expr_as_str, parseAll=parse_all
        )[0]