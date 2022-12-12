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
from . import semantics

statement_classifier = util.parsing.StatementClassifier()
semantics_checker = semantics.Semantics()

def _is_ignored_statement(tokens):
    """All statements beginning with the tokens below are ignored.
    """
    return (tokens[0].lower() in ["write","print","character","use","implicit","parameter"]
           or statement_classifier.is_declaration(tokens))

def _is_ignored_fortran_directive(tokens):
    return (util.parsing.compare_ignore_case(tokens[1:4],["acc","end","serial"])
           or util.parsing.compare_ignore_case(tokens[1:4],["acc","end","kernels"])
           or util.parsing.compare_ignore_case(tokens[1:4],["acc","end","parallel"])
           or util.parsing.compare_ignore_case(tokens[1:4],["acc","end","loop"])
           or util.parsing.compare_ignore_case(tokens[1:3],["acc","routine"]))

# todo resolve all at the end?
def _parse_arith_expr(expr_as_str,scope):
    parse_result = parse_arith_expr(expr_as_str)
    if scope != None:
        semantics_checker.resolve_arith_expr(parse_result,scope)
    return parse_result

def _parse_assignment(expr_as_str,scope):
    parse_result = parse_assignment(expr_as_str)
    if scope != None:
        semantics_checker.resolve_assignment(parse_result,scope)
    return parse_result

def _parse_acc_directive(expr_as_str,scope):
    parse_result = parse_acc_directive(expr_as_str)
    if scope != None:
        semantics_checker.resolve_acc_directive(parse_result,scope)
    return parse_result

@util.logging.log_entry_and_exit(opts.log_prefix)
def parse_fortran_code(code,result_name=None,scope=None):
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
    - parse other statements via pyparsing tree grammar
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
            "{}: {}".format(stmt_info.loc_str(), msg))

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
    def append_(node, kind=None, quiet=False):
        nonlocal stmt_info
        nonlocal curr
        nonlocal level
        if isinstance(node, tree.TTNode):
            node.parent = curr
        curr.body.append(node)
        if kind != None and not quiet:
            util.logging.log_debug2(
                opts.log_prefix, "parse_fortran_code.append_",
                "{0}: '{2}' -- found: {1}".format(
                  stmt_info.loc_str(),
                  kind, stmt))

    def descend_(container,
                 kind,
                 named_label=None,
                 inc_level=True):
        nonlocal stmt_info
        nonlocal curr
        nonlocal stmt
        nonlocal level
        assert isinstance(curr,tree.TTContainer)
        append_(container)
        if named_label != None: # append exit label behind construct
            exit_label = tree.TTLabel([named_label])
            exit_label.is_exit_marker = True
            exit_label.is_standalone = True
            append_(exit_label)
        curr = container
        curr.named_label = named_label
        if inc_level:
            level += 1
        util.logging.log_debug2(
            opts.log_prefix, "parse_fortran_code.append_",
            "{0}: '{2}' -- enter: {1}".format(
              stmt_info.loc_str(),
              kind, stmt))

    def ascend_(kind,named_label=None):
        nonlocal stmt_info
        nonlocal curr
        nonlocal stmt
        nonlocal level
        assert isinstance(curr,tree.TTContainer)
        if not curr.compare_label(curr.named_label,named_label):
           if curr.named_label == None:
               raise_syntax_error_(
                 "did not expect named label '{}'".format(named_label)
               )
           elif curr.named_label.isidentifier():
               raise_syntax_error_(
                 "expected named label '{}'".format(curr.named_label)
               )
        curr = curr.parent # must happen after label check
        level = min(level - 1, 0)
        util.logging.log_debug2(
            opts.log_prefix, "parse_fortran_code.append_",
            "{0}: '{2}' -- leave: {1}".format(
              stmt_info.loc_str(),
              kind, stmt))

    def ascend_from_do_loop_(kind,named_label=None):
        nonlocal curr
        saved = curr
        ascend_(kind, named_label)
        if isinstance(saved.parent,tree.TTAccLoop):
            ascend_("acc " + saved.parent.kind)

    # parser loop
    ttree = tree.TTRoot()
    curr = ttree
    level = 0
    for stmt_info in linemapper.statement_iterator(linemaps):
        stmt1 = stmt_info.body
        tokens = util.parsing.tokenize(stmt1.lower(), padded_size=1)
        if stmt_info.numeric_label != None:
            append_(tree.TTLabel([stmt_info.numeric_label]))
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
        if statement_classifier.is_blank_line(stmt):
            if type(curr) != tree.TTRoot:
                append_(tree.TTBlank([stmt]), "blank line")
        elif _is_ignored_statement(tokens):
            ignore_("statement")
        elif statement_classifier.is_fortran_directive(stmt,modern_fortran):
            try:
                append_(tree.TTCommentedOut([stmt]), "comment", quiet=True)
                if _is_ignored_fortran_directive(tokens):
                    ignore_("directive")
                elif (
                    util.parsing.compare_ignore_case(tokens[1:4],["acc","kernels","loop"])
                    or util.parsing.compare_ignore_case(tokens[1:4],["acc","parallel","loop"])
                    or util.parsing.compare_ignore_case(tokens[1:4],["cuf","kernel","do"]) # TODO no acc directive
                    or util.parsing.compare_ignore_case(tokens[1:3],["acc","kernels"])
                    or util.parsing.compare_ignore_case(tokens[1:3],["acc","parallel"])
                    or util.parsing.compare_ignore_case(tokens[1:3],["acc","serial"])
                    or util.parsing.compare_ignore_case(tokens[1:3],["acc","loop"])
                  ):
                    directive = _parse_acc_directive(stmt,scope)
                    descend_(
                      directive,
                      "acc " + directive.kind 
                    )
                else:
                    warn_("directive")
            except pyparsing.ParseException as e:
                error_("directive", e)
        elif statement_classifier.is_fortran_comment(stmt,modern_fortran):
            if type(curr) != tree.TTRoot:
                comment = re.split("!|^[c*]", stmt1, 1, re.IGNORECASE)[1]
                append_(tree.TTCommentedOut([comment]), "comment")
        elif statement_classifier.is_block(tokens):
            named_label = statement_classifier.parse_result
            descend_(
              tree.TTBlock([[]]),
              "block", 
              named_label=named_label
            )
        elif statement_classifier.is_do(tokens):
            (named_label, numeric_do_label, cond, var, 
            lbound_str, ubound_str, stride_str) = statement_classifier.parse_result
            if cond != None:
                do_loop = tree.TTDoWhile([
                  _parse_arith_expr(cond,scope),
                  []
                ])
            elif var != None: 
                try:
                    begin, end, stride = None, None, None
                    if var != None and lbound_str != None:
                        begin = _parse_assignment(
                          var + "=" + lbound_str,scope)
                    if ubound_str != None:
                        end = _parse_arith_expr(ubound_str,scope)
                    if stride_str != None and len(stride_str):
                        stride = _parse_arith_expr(stride_str,scope)
                    do_loop = tree.TTDo([begin, end, stride, []])
                except pyparsing.ParseException as e:
                    error_("do loop", e)
            else:
                do_loop = tree.TTUnconditionalDo([[]])
            do_loop.numeric_do_label = numeric_do_label
            descend_(
              do_loop,
              "do loop", 
              named_label=named_label
            )
            if named_label: # put into body of do loop
                do_loop_body_label = tree.TTLabel([named_label])
                do_loop_body_label.is_standalone = True
                append_(do_loop_body_label)
        # if-then-else
        elif statement_classifier.is_if_then(tokens):
            (named_label, cond) = statement_classifier.parse_result
            descend_(
              tree.TTIfElseBlock(), "if block", 
              named_label=named_label,
              inc_level=False
            )
            try:
                if_branch = tree.TTIfElseIf([
                  None,
                  _parse_arith_expr(cond,scope),
                  []
                ])
            except pyparsing.ParseException as e:
                error_("if", e)
            descend_(
              if_branch,
              "if branch"
            )
        elif statement_classifier.is_else_if_then(tokens):
            assert type(curr) is tree.TTIfElseIf
            ascend_("if")
            cond = statement_classifier.parse_result
            try:
                else_if_branch = tree.TTIfElseIf([
                  "else",
                  _parse_arith_expr(cond,scope),
                  []
                ])
                descend_(
                  else_if_branch,
                  "else-if branch"
                )
            except pyparsing.ParseException as e:
                error_("else-if", e)
        elif statement_classifier.is_else(tokens):
            cond = statement_classifier.parse_result
            assert type(curr) is tree.TTIfElseIf, type(curr)
            ascend_("if/else-if")
            descend_(
              tree.TTElse([]),
              "else branch"
            )
        # select-case
        elif statement_classifier.is_select_case(tokens):
            (named_label, argument) = statement_classifier.parse_result
            try:
                descend_(
                  tree.TTSelectCase([
                    _parse_arith_expr(named_label,scope),
                    []
                  ]),
                  "select-case",
                  named_label=named_label
                )
            except pyparsing.ParseException as e:
                error_("select-case", e)
        elif statement_classifier.is_case(tokens):
            values = statement_classifier.parse_result
            if type(curr) is tree.TTCase:
                ascend_("case")
            try:
                descend_(
                  tree.TTCase([
                    ttvalues, 
                    [_parse_arith_expr(v,scope) for v in values]
                  ]), "case"
                )
            except pyparsing.ParseException as e:
                error_("case", e)
        elif statement_classifier.is_case_default(tokens):
            if type(curr) is tree.TTCase:
                ascend_("case")
            descend_(tree.TTCaseDefault([]), "case default")
        # end
        elif tokens[0] == "continue":
            ttcontinue = tree.TTContinue([[]])
            ttcontinue._result_name = result_name
            append_(ttcontinue,"continue statement")
            if stmt_label != None and stmt_label.isnumeric():
                while ( isinstance(curr,tree.TTDo)
                        and curr.compare_dolabel(stmt_label) ):
                    ascend_from_do_loop_(tokens[0], named_label=named_label)
        elif statement_classifier.is_cycle(tokens):
            # todo: might have label arg
            ttcycle = tree.TTCycle([statement_classifier.parse_result])
            append_(ttcycle,"cycle statement")
        elif statement_classifier.is_exit(tokens):
            # todo: might have label arg
            ttexit = tree.TTExit([statement_classifier.parse_result])
            append_(ttexit,"exit statement")
        elif statement_classifier.is_unconditional_goto(tokens):
            # todo: consider assign <numeric-label> to <ident>
            append_(tree.TTUnconditionalGoTo([statement_classifier.parse_result]),"goto statement")
        elif statement_classifier.is_assigned_goto(tokens):
            # todo: consider assign <numeric-label> to <ident>
            raise util.error.LimitationError("assigned goto statement not supported yet")
        elif statement_classifier.is_computed_goto(tokens):
            raise util.error.LimitationError("computed goto statement not supported yet")
        elif statement_classifier.is_end(tokens):
            assert isinstance(curr,tree.TTContainer)
            ascend_("structured block")
        elif statement_classifier.is_end(tokens,"block"):
            assert isinstance(curr,tree.TTBlock)
            named_label = statement_classifier.parse_result
            ascend_("executable block", named_label=named_label)
        elif statement_classifier.is_end(tokens,"do"):
            assert isinstance(curr,tree.TTDo), type(curr)
            named_label = statement_classifier.parse_result
            ascend_from_do_loop_("do", named_label=named_label)
        elif statement_classifier.is_end(tokens, "if"):
            named_label = statement_classifier.parse_result
            assert isinstance(curr,(tree.TTElse,tree.TTIfElseIf))
            ascend_("if/else-if/else branch") # branch
            assert isinstance(curr,tree.TTIfElseBlock)
            ascend_("if-else block", named_label=named_label) # block
        elif statement_classifier.is_end(tokens, "select"):
            named_label = statement_classifier.parse_result
            assert isinstance(curr,(tree.TTCaseDefault,tree.TTCase))
            ascend_("case/case default") # case
            assert isinstance(curr,tree.TTSelectCase)
            ascend_("select case", named_label=named_label) # block
        # single statements
        elif statement_classifier.is_pointer_assignment(tokens):
            error_("pointer assignment")
        elif statement_classifier.is_assignment(tokens):
            try:
                assignment_variant = _parse_assignment(
                    stmt, scope) 
                append_(assignment_variant, "assignment")
            except pyparsing.ParseException as e:
                error_("assignment", e)
        elif tokens[0] == "return":
            ttreturn = tree.TTReturn([])
            ttreturn._result_name = result_name
            append_(ttreturn,"return statement")
        elif statement_classifier.is_subroutine_call(tokens):
            try:
                parse_result = tree.grammar.fortran_subroutine_call.parseString(
                    stmt, parseAll=True)
                append_(parse_result[0], "subroutine call")
            except pyparsing.ParseException as e:
                error_("subroutine call", e)
        else:
            error_("unknown and not ignored")

    return ttree
    
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

# API
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
        return tree.grammar_no_logic_no_custom.arith_expr.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    elif not contains_logic_ops:
        return tree.grammar_no_logic.arith_expr.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    elif not contains_custom_ops:
        return tree.grammar_no_custom.arith_expr.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    else:
        return tree.grammar.arith_expr.parseString(
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
        return tree.grammar_no_logic_no_custom.assignment.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    elif not contains_logic_ops:
        return tree.grammar_no_logic.assignment.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    elif not contains_custom_ops:
        return tree.grammar_no_custom.assignment.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    else:
        return tree.grammar.assignment.parseString(
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
        return tree.grammar_no_logic_no_custom.rvalue.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    elif not contains_logic_ops:
        return tree.grammar_no_logic.rvalue.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    elif not contains_custom_ops:
        return tree.grammar_no_custom.rvalue.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    else:
        return tree.grammar.rvalue.parseString(
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
        return tree.grammar_no_logic_no_custom.acc_directive.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    elif not contains_logic_ops:
        return tree.grammar_no_logic.acc_directive.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    elif not contains_custom_ops:
        return tree.grammar_no_custom.acc_directive.parseString(
          expr_as_str, parseAll=parse_all
        )[0]
    else:
        return tree.grammar.acc_directive.parseString(
          expr_as_str, parseAll=parse_all
        )[0]

# API
# todo: parsing and translation is similar but analysis differs between the different kernel
# types. For example for CUF, the reduction vars must be detected by the parser (lhs scalars)
# while they are specified with ACC,OMP.


#def parse_compute_construct(fortran_statements, scope=None):
#    """:return: C snippet equivalent to original Fortran code.
#    """
#    ttcomputeconstruct = parse_fortran_code(fortran_statements).body[0]
#
#    ttcomputeconstruct.scope = scope
#    return ttcomputeconstruct
#
#
#def parse_procedure_body(fortran_statements, scope=None, result_name=None):
#    """Parse a function/subroutine body.
#    """
#    parse_result = parse_fortran_code(fortran_statements,result_name)
#    ttprocedurebody = tree.TTProcedureBody([parse_result.body])
#
#    ttprocedurebody.scope = scope
#    ttprocedurebody.result_name = result_name
#    return ttprocedurebody
