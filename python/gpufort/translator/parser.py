# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os
import os, sys, traceback
import logging
import collections
import ast
import re

import pyparsing

from gpufort import util

from . import tree
from . import opts
from . import conv
from . import prepostprocess

@util.logging.log_entry_and_exit(opts.log_prefix)
def parse_fortran_code(statements,result_name=None):
    """
    :param list statements: list of single-line statements in lower case
    :param result_name: Return value to use for any return statements found
                        in the statements. If set to None, this assumes
                        that we currently parse the body of a subroutine, i.e.
                        a void function in C.
    
    requirements:
    
    - input is statements:
      - all lower case
      - & multiline statement/directives have been converted to single-line statement
      - single-line multi-statement has been converted to multiple single statements
      - lines may contain comments and directives appended to actual code
        - comments must be preserved
  
    behavior:
    - treat do, do while, if, elseif, else, select and end statements explicitly
    - parse other statements
    - consider comments
    - every call returns a subtree
    """
    modern_fortran = opts.modern_fortran 

    # tree creation ops
    def append_(node, kind=None):
        nonlocal curr
        nonlocal level
        if isinstance(node, tree.TTNode):
            node.parent = curr
        curr.body.append(node)
        if kind != None:
            util.logging.log_debug2(
                opts.log_prefix, "parse_fortran_code.append_",
                "found {} in statement '{}'".format(kind, stmt))

    def descend_(node, kind, inc_level=True):
        nonlocal curr
        nonlocal stmt
        nonlocal level
        append_(node)
        curr = node
        if inc_level:
            level += 1
        util.logging.log_debug2(
            opts.log_prefix, "parse_fortran_code.append_",
            "enter {} in statement '{}'".format(kind, stmt))

    def ascend_(kind):
        nonlocal curr
        nonlocal stmt
        nonlocal level
        curr = curr.parent
        level = min(level - 1, 0)
        util.logging.log_debug2(
            opts.log_prefix, "parse_fortran_code.append_",
            "leave {} in statement '{}'".format(kind, stmt))

    # error handling
    def error_(expr, exception=None):
        nonlocal stmt1
        if exception != None:
            debug_msg = ": " + str(exception)
            util.logging.log_debug(opts.log_prefix,
                                   "parse_fortran_code", debug_msg)
        raise util.error.SyntaxError(
            "failed to parse {} expression '{}'".format(expr, stmt))

    def warn_(expr, exception=None):
        nonlocal stmt1
        util.logging.log_warning(opts.log_prefix, "parse_fortran_code",
                              "ignored {} expression '{}'".format(expr, stmt))
        if exception != None:
            util.logging.log_debug(opts.log_prefix,
                                   "parse_fortran_code",
                                   str(exception))

    def ignore_(expr):
        nonlocal stmt1
        util.logging.log_debug3(opts.log_prefix, "parse_fortran_code",
                                "ignored {} '{}'".format(expr, stmt))

    # parser loop
    ttree = tree.TTRoot()
    curr = ttree
    curr_offload_region = None
    curr_offload_loop = None
    level = 0
    for stmt1 in statements:
        tokens = util.parsing.tokenize(stmt1.lower(), padded_size=6)
        stmt = " ".join(tokens) # ensure whitespace between operators and operands
        stmt = prepostprocess.preprocess_fortran_statement(stmt)
        # strip of ! from tokens.index("!")
        if "!" in stmt:
            stmt_no_comment = stmt.split("!")[0].lower()
        else:
            stmt_no_comment = stmt.lower()
        util.logging.log_debug3(
            opts.log_prefix, "parse_fortran_code",
            "process statement '{}' (preprocessed: '{}')".format(stmt1, stmt))
        if len(tokens):
            util.logging.log_debug4(opts.log_prefix,
                                    "parse_fortran_code",
                                    "tokens=['{}']".format("','".join(tokens)))
        # tree construction
        if util.parsing.is_blank_line(stmt):
            if type(curr) != tree.TTRoot:
                append_(stmt, "blank line")
        elif util.parsing.is_ignored_statement(tokens):
            ignore_("statement")
        elif util.parsing.is_fortran_directive(stmt,modern_fortran):
            try:
                if util.parsing.is_ignored_fortran_directive(tokens):
                    ignore_("directive")
                elif util.parsing.is_fortran_offload_region_plus_loop_directive(
                        tokens): # most complex first
                    parse_result = tree.grammar.loop_annotation.parseString(
                        stmt, parseAll=True)
                    curr_offload_region = parse_result[0]
                    curr_offload_loop = parse_result[0]
                    descend_(tree.TTComputeConstruct(stmt_no_comment,"", [curr_offload_region, []]),\
                            "compute construct plus loop directive")
                elif util.parsing.is_fortran_offload_region_directive(tokens):
                    parse_result = tree.grammar.offload_region_start.parseString(
                        stmt, parseAll=True)
                    curr_offload_region = parse_result[0]
                    descend_(tree.TTComputeConstruct(stmt_no_comment,"", [curr_offload_region, []]),\
                            "compute construct")
                elif util.parsing.is_fortran_offload_loop_directive(tokens):
                    parse_result = tree.grammar.loop_annotation.parseString(
                        stmt, parseAll=True)
                    curr_offload_loop = parse_result[0]
                    util.logging.log_debug2(
                        opts.log_prefix, "parse_fortran_code.append_",
                        "found {} in statement '{}'".format(
                            "loop directive", stmt))
                else:
                    warn_("directive")
            except pyparsing.ParseException as e:
                error_("directive", e)
                pass
        elif util.parsing.is_fortran_comment(stmt,modern_fortran):
            if type(curr) != tree.TTRoot:
                comment = re.split("!|^[c*]", stmt1, 1, re.IGNORECASE)[1]
                append_("// " + comment + "\n", "comment")
        # do/while
        elif util.parsing.is_do_while(tokens):
            try:
                parse_result = tree.grammar.fortran_do_while.parseString(
                    stmt_no_comment, parseAll=True)
                descend_(tree.TTDoWhile(stmt, 0,
                                        parse_result.asList() + [[]]),
                         "do-while loop")
            except pyparsing.ParseException as e:
                error_("do-while loop", e)
        elif util.parsing.is_do(tokens):
            result = util.parsing.parse_do_statement(stmt_no_comment)
            label, var, lbound_str, ubound_str, stride_str = result
            begin, end, stride = None, None, None
            try:
                begin = tree.grammar.assignment.parseString(
                        "".join([var,"=",lbound_str]))[0]
            except pyparsing.ParseException as e:
                error_("do loop: begin", e)
            try:
                end = tree.grammar.arithmetic_expression.parseString(ubound_str)[0]
            except pyparsing.ParseException as e:
                error_("do loop: end", e)
            if stride_str != None and len(stride_str):
                try:
                    stride = tree.grammar.arithmetic_expression.parseString(stride_str)[0]
                except pyparsing.ParseException as e:
                    error_("do loop: stride", e)
            do_loop = tree.TTDo(stmt, 0, [curr_offload_loop, begin, end, stride, []])
            descend_(do_loop, "do loop")
            curr_offload_loop = None
        # if-then-else
        elif util.parsing.is_if_then(tokens):
            try:
                descend_(tree.TTIfElseBlock(), "if block", inc_level=False)
                parse_result = tree.grammar.fortran_if_else_if.parseString(
                    stmt_no_comment, parseAll=True)
                descend_(
                    tree.TTIfElseIf(stmt_no_comment, 0,
                                    parse_result.asList() + [[]]), "if branch")
            except pyparsing.ParseException as e:
                error_("if", e)
        elif util.parsing.is_else_if_then(tokens):
            assert type(curr) is tree.TTIfElseIf
            ascend_("if")
            try:
                parse_result = tree.grammar.fortran_if_else_if.parseString(
                    stmt_no_comment, parseAll=True)
                #print(parse_result)
                descend_(
                    tree.TTIfElseIf(stmt_no_comment, 0,
                                    parse_result.asList() + [[]]),
                    "else-if branch")
            except pyparsing.ParseException as e:
                error_("else-if", e)
        elif util.parsing.is_else(tokens):
            assert type(curr) is tree.TTIfElseIf, type(curr)
            ascend_("if/else-if")
            descend_(tree.TTElse(stmt_no_comment, 0, []), "else branch")
        # select-case
        elif util.parsing.is_select_case(tokens):
            try:
                parse_result = tree.grammar.fortran_select_case.parseString(
                    stmt_no_comment, parseAll=True)
                descend_(
                    tree.TTSelectCase(stmt_no_comment, 0, parse_result.asList() + [[]]),
                    "select-case")
            except pyparsing.ParseException as e:
                error_("select-case", e)
        elif util.parsing.is_case(tokens):
            if type(curr) is tree.TTCase:
                ascend_("case")
            try:
                parse_result = tree.grammar.fortran_case.parseString(
                    stmt_no_comment, parseAll=True)
                descend_(
                    tree.TTCase(stmt_no_comment, 0,
                                parse_result.asList() + [[]]), "case")
            except pyparsing.ParseException as e:
                error_("case", e)
        elif util.parsing.is_case_default(tokens):
            if type(curr) is tree.TTCase:
                ascend_("case-default")
            descend_(tree.TTCaseDefault(stmt_no_comment, 0, [[]]), "case default")
        # end
        elif util.parsing.is_end(tokens, ["do"]):
            ascend_(tokens[1])
        elif util.parsing.is_end(tokens, ["if", "select"]):
            ascend_(tokens[1])
            ascend_(tokens[1])
        # single statements
        elif util.parsing.is_pointer_assignment(tokens):
            error_("pointer assignment")
        elif util.parsing.is_assignment(tokens):
            try:
                assignment_variant = tree.grammar.fortran_assignment.parseString(
                    stmt_no_comment, parseAll=True)[0]
                append_(parse_result[0], "assignment")
            except pyparsing.ParseException as e:
                error_("assignment", e)
        elif tokens[0] == "return":
            ttreturn = tree.TTReturn(stmt_no_comment, 0, [])
            ttreturn._result_name = result_name
            append_(ttreturn,"return statement")
        elif util.parsing.is_subroutine_call(tokens):
            try:
                parse_result = tree.grammar.fortran_subroutine_call.parseString(
                    stmt_no_comment, parseAll=True)
                append_(parse_result[0], "subroutine call")
            except pyparsing.ParseException as e:
                error_("subroutine call", e)
        else:
            error_("unknown and not ignored")

    return ttree


# API
def convert_arithmetic_expression(fortran_snippet):
    return (matrix_arithmetic_expression | complex_arithmetic_expression
            | arithmetic_expression).parseString(fortran_snippet)[0].c_str()


def parse_attributes(ttattributes):
    attribute = tree.make_f_str(ttattributes.qualifiers[0]).lower()
    modified_vars = [tree.make_f_str(var).lower() for var in ttattributes._rhs]
    return attribute, modified_vars


# TODO parsing and translation is similar but analysis differs between the different kernel
# types. For example for CUF, the reduction vars must be detected by the parser (lhs scalars)
# while they are specified with ACC,OMP.


def parse_loopnest(fortran_statements, scope=None):
    """:return: C snippet equivalent to original Fortran code.
    """
    ttloopnest = parse_fortran_code(fortran_statements).body[0]
    curr_offload_region = None

    ttloopnest.scope = scope
    return ttloopnest


def parse_procedure_body(fortran_statements, scope=None, result_name=None):
    """Parse a function/subroutine body.
    """
    parse_result = parse_fortran_code(fortran_statements,result_name)
    ttprocedurebody = tree.TTProcedureBody("", 0, [parse_result.body])

    ttprocedurebody.scope = scope
    ttprocedurebody.result_name = result_name
    return ttprocedurebody
