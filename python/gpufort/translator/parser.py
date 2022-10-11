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
from gpufort import linemapper

from . import tree
from . import opts
from . import conv
from . import prepostprocess

def _is_ignored_statement(tokens):
    """All statements beginning with the tokens below are ignored.
    """
    return (tokens[0].lower() in ["write","print","character","use","implicit","parameter"]
           or util.parsing.is_declaration(tokens))

def _is_ignored_fortran_directive(tokens):
    return (util.parsing.compare_ignore_case(tokens[1:4],["acc","end","serial"])
           or util.parsing.compare_ignore_case(tokens[1:4],["acc","end","kernels"])
           or util.parsing.compare_ignore_case(tokens[1:4],["acc","end","parallel"])
           or util.parsing.compare_ignore_case(tokens[1:4],["acc","end","loop"])
           or util.parsing.compare_ignore_case(tokens[1:3],["acc","routine"]))

@util.logging.log_entry_and_exit(opts.log_prefix)
def parse_fortran_code(code,result_name=None):
    """
    :param code: text, list of lines, or list of linemaps, see gpufort.linemapper component.
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
    linemaps = None
    if type(code) == str:
        if len(code):
            linemaps = linemapper.read_lines(code.splitlines(keep=True))
        else:
            return tree.TTRoot()
    elif type(code) == list:
        if len(code) > 0:
            if type(code[0]) == str:
                linemaps = linemapper.read_lines(code)
            else:
                linemaps = code
        else:
            return tree.TTRoot()
    assert linemaps != None, "argument 'code' must be str, list of str, or list of linemaps, see gpufort.linemapper component" 

    # todo: identifier labels are case senstivive -> Catch labels via the linemapper
    # Make parser use linemaps
    modern_fortran = opts.modern_fortran 


    # error handling
    def raise_syntax_error_(msg):
        nonlocal stmt_info
        raise util.error.SyntaxError(
            "{}: {}'".format(stmt_info.loc_str(), msg))

    def error_(expr, exception=None):
        nonlocal stmt_info
        nonlocal stmt1
        if exception != None:
            debug_msg = ": " + stmt_info.loc_str() + ":" + str(exception)
            util.logging.log_debug(opts.log_prefix,
                                   "parse_fortran_code", debug_msg)
        raise_syntax_error_(
          "failed to parse {} expression '{}'".format(expr, stmt)
        )

    def warn_(expr, exception=None):
        nonlocal stmt_info
        nonlocal stmt1
        util.logging.log_warning(opts.log_prefix, "parse_fortran_code",
                              "{}: ignored {} expression '{}'".format(stmt_info.loc_str(),expr, stmt))
        if exception != None:
            util.logging.log_debug(opts.log_prefix,
                                   "parse_fortran_code",
                                   stmt_info.loc_str()+":"+str(exception))

    def ignore_(expr):
        nonlocal stmt_info
        nonlocal stmt1
        util.logging.log_debug3(opts.log_prefix, "parse_fortran_code",
                                "{}: ignored {} '{}'".format(stmt_info.loc_str(),expr, stmt))
    
    # tree creation ops
    def append_(node, kind=None):
        nonlocal stmt_info
        nonlocal curr
        nonlocal level
        if isinstance(node, tree.TTNode):
            node.parent = curr
        curr.body.append(node)
        if kind != None:
            util.logging.log_debug2(
                opts.log_prefix, "parse_fortran_code.append_",
                "{}: found {} in statement '{}'".format(
                  stmt_info.loc_str(),
                  kind, stmt))

    def descend_(container,
                 kind,
                 inc_level=True):
        nonlocal stmt_info
        nonlocal curr
        nonlocal stmt
        nonlocal level
        assert isinstance(curr,tree.TTContainer)
        append_(container)
        curr = container
        curr.label = label
        if inc_level:
            level += 1
        util.logging.log_debug2(
            opts.log_prefix, "parse_fortran_code.append_",
            "{}: enter {} in statement '{}'".format(
              stmt_info.loc_str(),
              kind, stmt))

    def ascend_(kind):
        nonlocal stmt_info
        nonlocal curr
        nonlocal stmt
        nonlocal level
        assert isinstance(curr.parent,tree.TTContainer)
        if not curr.parent.compare_label(label):
           print("labels") 
           print(label)
           print(curr.parent.label)
           if curr.parent.label == None:
               raise_syntax_error_(
                 "did not expect label"
               )
           elif curr.parent.label.isidentifier():
               raise_syntax_error_(
                 "expected named label '{}'".format(curr.parent.label)
               )
           elif curr.parent.label.isnumeric():
               raise_syntax_error_(
                 "expected numeric label '{}'".format(curr.parent.label)
               )
        curr = curr.parent
        level = min(level - 1, 0)
        util.logging.log_debug2(
            opts.log_prefix, "parse_fortran_code.append_",
            "{}: leave {} in statement '{}'".format(
              stmt_info.loc_str(),
              kind, stmt))

    # parser loop
    ttree = tree.TTRoot()
    curr = ttree
    level = 0
    for stmt_info in linemapper.statement_iterator(linemaps):
        stmt1 = stmt_info.body
        stmt_numeric_label = stmt_info.numeric_label
        stmt_named_label = stmt_info.named_label
        tokens = util.parsing.tokenize(stmt1.lower(), padded_size=1)
        if stmt_label != None:
            append_(tree.TTLabel([stmt_label]))
        stmt = prepostprocess.preprocess_fortran_statement(tokens)
        tokens = util.parsing.tokenize(stmt,padded_size=6)
        # strip of ! from tokens.index("!")
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
                append_(tree.TTBlank([stmt]), "blank line")
        elif _is_ignored_statement(tokens):
            ignore_("statement")
        elif util.parsing.is_fortran_directive(stmt,modern_fortran):
            try:
                append_(tree.TTCommentedOut([stmt]), "comment")
                if _is_ignored_fortran_directive(tokens):
                    ignore_("directive")
                elif util.parsing.compare_ignore_case(tokens[1:4],["acc","kernels","loop"]):
                    append_(
                      tree.grammar.acc_parallel_loop.parseString(
                        stmt, parseAll=True)[0],
                      "acc parallel loop construct"
                    )
                elif util.parsing.compare_ignore_case(tokens[1:3],["acc","kernels"]):
                    append_(
                      tree.grammar.acc_kernels.parseString(
                        stmt, parseAll=True)[0],
                      "acc kernels construct"
                    )
                elif util.parsing.compare_ignore_case(tokens[1:4],["acc","parallel","loop"]):
                    append_(
                      tree.grammar.acc_parallel_loop.parseString(
                        stmt, parseAll=True)[0],
                      "acc parallel loop construct"
                    )
                elif util.parsing.compare_ignore_case(tokens[1:3],["acc","parallel"]):
                    append_(
                      tree.grammar.acc_parallel.parseString(
                        stmt, parseAll=True)[0],
                      "acc parallel construct"
                    )
                elif util.parsing.compare_ignore_case(tokens[1:3],["acc","serial"]):
                    append_(
                      tree.grammar.acc_serial.parseString(
                        stmt, parseAll=True)[0],
                      "acc serial construct"
                    )
                elif util.parsing.compare_ignore_case(tokens[1:3],["acc","loop"]):
                    append_(
                      tree.grammar.acc_loop.parseString(
                        stmt, parseAll=True)[0],
                      "acc loop directive"
                    )
                elif util.parsing.compare_ignore_case(tokens[1:4],["cuf","kernel","do"]):
                    append_(
                      tree.grammar.cuf_kernel_do.parseString(
                        stmt, parseAll=True)[0],
                      "cuf kernel do construct"
                    )
                else:
                    warn_("directive")
            except pyparsing.ParseException as e:
                error_("directive", e)
                pass
        elif util.parsing.is_fortran_comment(stmt,modern_fortran):
            if type(curr) != tree.TTRoot:
                comment = re.split("!|^[c*]", stmt1, 1, re.IGNORECASE)[1]
                append_(tree.TTCommentedOut([comment]), "comment")
        # do/while
        elif util.parsing.is_do_while(tokens):
            # todo: parse do while with numeric label, do <numeric-lable> while ( <cond> ) ...
            try:
                parse_result = tree.grammar.fortran_do_while.parseString(
                    stmt, parseAll=True)
                descend_(tree.TTDoWhile(
                  parse_result.asList() + [[]]),
                  "do-while loop",
                  numeric_label=numeric_label,
                  named_label=named_label)
            except pyparsing.ParseException as e:
                error_("do-while loop", e)
        elif util.parsing.is_do(tokens):
            # todo: parse do without arguments, do <numeric-label> ... <numeric-label> ( <cond> ) ...
            result = util.parsing.parse_do_statement(stmt)
            numeric_do_loop_label, var, lbound_str, ubound_str, stride_str = result
            # Legal:
            # outer: do i = ...
            # end do outer
            #
            # do <int-1> i = ...
            # <int-1> continue
            #
            # <int-2> do <int-1> i = ...
            # <int-1> continue
            if numeric_do_loop_label != None and named_label != None and named_label.isidentifier():
                raise raise_syntax_error_("cannot specify numeric and named label together")
            #
            begin, end, stride = None, None, None
            try:
                if var != None and lbound_str != None:
                    begin = tree.grammar.assignment.parseString(var + "=" + lbound_str)[0]
            except pyparsing.ParseException as e:
                error_("do loop: begin", e)
            try:
                if ubound_str != None:
                    end = tree.grammar.arith_logic_expr.parseString(ubound_str)[0]
            except pyparsing.ParseException as e:
                error_("do loop: end", e)
            if stride_str != None and len(stride_str):
                try:
                    stride = tree.grammar.arith_logic_expr.parseString(stride_str)[0]
                except pyparsing.ParseException as e:
                    error_("do loop: stride", e)
            do_loop = tree.TTDo([begin, end, stride, []])
            do_loop.numeric_do_loop_label = numeric_do_loop_label
            descend_(do_loop, "do loop")
        # if-then-else
        elif util.parsing.is_if_then(tokens):
            try:
                descend_(tree.TTIfElseBlock(), "if block", stmt_label, inc_level=False)
                parse_result = tree.grammar.fortran_if_else_if.parseString(
                    stmt, parseAll=True)
                descend_(
                    tree.TTIfElseIf(
                                    parse_result.asList() + [[]]), "if branch")
            except pyparsing.ParseException as e:
                error_("if", e)
        elif util.parsing.is_else_if_then(tokens):
            assert type(curr) is tree.TTIfElseIf
            ascend_("if")
            try:
                parse_result = tree.grammar.fortran_if_else_if.parseString(
                    stmt, parseAll=True)
                #print(parse_result)
                descend_(
                    tree.TTIfElseIf(
                                    parse_result.asList() + [[]]),
                    "else-if branch")
            except pyparsing.ParseException as e:
                error_("else-if", e)
        elif util.parsing.is_else(tokens):
            assert type(curr) is tree.TTIfElseIf, type(curr)
            ascend_("if/else-if")
            descend_(tree.TTElse([]), "else branch")
        # select-case
        elif util.parsing.is_select_case(tokens):
            try:
                parse_result = tree.grammar.fortran_select_case.parseString(
                    stmt, parseAll=True)
                descend_(
                    tree.TTSelectCase(parse_result.asList() + [[]]),
                    "select-case",
                    stmt_label
                )
            except pyparsing.ParseException as e:
                error_("select-case", e)
        elif util.parsing.is_case(tokens):
            if type(curr) is tree.TTCase:
                ascend_("case")
            try:
                values = util.parsing.parse_case_statement(tokens)
                descend_(
                    tree.TTCase(
                                [values, []]), "case")
            except pyparsing.ParseException as e:
                error_("case", e)
        elif util.parsing.is_case_default(tokens):
            if type(curr) is tree.TTCase:
                ascend_("case-default")
            descend_(tree.TTCaseDefault([[]]), "case default")
        # end
        elif util.parsing.is_end(tokens, ["do"]):
            assert isinstance(curr,tree.TTDo)
            ascend_(tokens[1], stmt_label)
        elif tokens[0] == "continue":
            ttcontinue = tree.TTContinue([[]])
            ttcontinue._result_name = result_name
            append_(ttcontinue,"continue statement")
            if stmt_label != None and stmt_label.isnumeric():
                while ( isinstance(curr,tree.TTDo)
                        and curr.compare_label(stmt_label) ):
                    ascend_(tokens[1], stmt_label)
        elif tokens[0] == "cycle":
            ttcontinue = tree.TTCycle([[]])
            append_(ttcontinue,"cycle statement")
        elif tokens[0] == "exit":
            ttexit = tree.TTExit([[]])
            append_(ttexit,"exit statement")
        elif tokens[0:2] == ["go","to"]:
            append_(tree.TTGoto([tokens[2]]),"goto statement")
        elif util.parsing.is_end(tokens, ["if", "select"]):
            ascend_(tokens[1], stmt_label)
            ascend_(tokens[1], stmt_label)
        # single statements
        elif util.parsing.is_pointer_assignment(tokens):
            error_("pointer assignment")
        elif util.parsing.is_assignment(tokens):
            try:
                assignment_variant = tree.grammar.fortran_assignment.parseString(
                    stmt, parseAll=True)[0]
                append_(assignment_variant, "assignment")
            except pyparsing.ParseException as e:
                error_("assignment", e)
        elif tokens[0] == "return":
            ttreturn = tree.TTReturn([])
            ttreturn._result_name = result_name
            append_(ttreturn,"return statement")
        elif util.parsing.is_subroutine_call(tokens):
            try:
                parse_result = tree.grammar.fortran_subroutine_call.parseString(
                    stmt, parseAll=True)
                append_(parse_result[0], "subroutine call")
            except pyparsing.ParseException as e:
                error_("subroutine call", e)
        else:
            error_("unknown and not ignored")

    return ttree

# API
def convert_arith_expr(fortran_snippet):
    return (matrix_arith_expr | complex_arith_expr
            | arith_logic_expr).parseString(fortran_snippet)[0].cstr()


def parse_attributes(ttattributes):
    attribute = tree.make_fstr(ttattributes.qualifiers[0]).lower()
    modified_vars = [tree.make_fstr(var).lower() for var in ttattributes._rhs]
    return attribute, modified_vars

# todo: parsing and translation is similar but analysis differs between the different kernel
# types. For example for CUF, the reduction vars must be detected by the parser (lhs scalars)
# while they are specified with ACC,OMP.


def parse_compute_construct(fortran_statements, scope=None):
    """:return: C snippet equivalent to original Fortran code.
    """
    ttcomputeconstruct = parse_fortran_code(fortran_statements).body[0]

    ttcomputeconstruct.scope = scope
    return ttcomputeconstruct


def parse_procedure_body(fortran_statements, scope=None, result_name=None):
    """Parse a function/subroutine body.
    """
    parse_result = parse_fortran_code(fortran_statements,result_name)
    ttprocedurebody = tree.TTProcedureBody([parse_result.body])

    ttprocedurebody.scope = scope
    ttprocedurebody.result_name = result_name
    return ttprocedurebody
