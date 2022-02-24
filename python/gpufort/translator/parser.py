# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os
import os, sys, traceback
import logging
import collections
import ast
import re
from gpufort import util

from . import tree
from . import opts
from . import conv
from . import prepostprocess


@util.logging.log_entry_and_exit(opts.log_prefix)
def _parse_fortran_code(statements, scope=None):
    """
    :param list for
    
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

    # tree creation ops
    def append_(node, kind=None):
        nonlocal curr
        nonlocal level
        if isinstance(node, tree.TTNode):
            node.parent = curr
            node.indent = "  " * level
        curr.body.append(node)
        if kind != None:
            util.logging.log_debug2(
                opts.log_prefix, "_parse_fortran_code.append_",
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
            opts.log_prefix, "_parse_fortran_code.append_",
            "enter {} in statement '{}'".format(kind, stmt))

    def ascend_(kind):
        nonlocal curr
        nonlocal stmt
        nonlocal level
        curr = curr.parent
        level = min(level - 1, 0)
        util.logging.log_debug2(
            opts.log_prefix, "_parse_fortran_code.append_",
            "leave {} in statement '{}'".format(kind, stmt))

    # error handling
    def error_(expr, exception=None):
        nonlocal stmt1
        util.logging.log_error(
            opts.log_prefix, "_parse_fortran_code",
            "failed to parse {} expression '{}'".format(expr, stmt))
        if exception != None:
            debug_msg = ": " + str(exception)
            util.logging.log_debug(opts.log_prefix,
                                   "_parse_fortran_code", debug_msg)
        sys.exit(2) # TODO error code

    def warn_(expr, exception=None):
        nonlocal stmt1
        util.logging.log_warn(opts.log_prefix, "_parse_fortran_code",
                              "ignored {} expression '{}'".format(expr, stmt))
        if exception != None:
            util.logging.log_debug(opts.log_prefix,
                                   "_parse_fortran_code",
                                   str(exception))
        sys.exit(2) # TODO error code

    def ignore_(expr):
        nonlocal stmt1
        util.logging.log_debug3(opts.log_prefix, "_parse_fortran_code",
                                "ignored {} '{}'".format(expr, stmt))

    # parser loop
    ttree = tree.TTRoot()
    curr = ttree
    curr_offload_region = None
    curr_offload_loop = None
    level = 0
    for stmt1 in statements:
        stmt = prepostprocess.preprocess_fortran_statement(stmt1)
        tokens = util.parsing.tokenize(stmt.lower(), padded_size=6)
        # strip of ! from tokens.index("!")
        if "!" in stmt:
            stmt_no_comment = stmt.split("!")[0].lower()
        else:
            stmt_no_comment = stmt.lower()
        util.logging.log_debug3(
            opts.log_prefix, "_parse_fortran_code",
            "process statement '{}' (preprocessed: '{}')".format(stmt1, stmt))
        if len(tokens):
            util.logging.log_debug4(opts.log_prefix,
                                    "_parse_fortran_code",
                                    "tokens=['{}']".format("','".join(tokens)))
        # tree construction
        if util.parsing.is_blank_line(stmt):
            if type(curr) != tree.TTRoot:
                append_(stmt, "blank line")
        elif util.parsing.is_ignored_statement(tokens):
            ignore_("statement")
        elif util.parsing.is_comment(tokens, stmt):
            if type(curr) != tree.TTRoot:
                comment = re.split("!|^[c*]", stmt1, 1, re.IGNORECASE)[1]
                append_("// " + comment + "\n", "comment")
        elif util.parsing.is_fortran_directive(tokens, stmt):
            try:
                if util.parsing.is_ignored_fortran_directive(tokens):
                    ignore_("directive")
                elif util.parsing.is_fortran_offload_region_plus_loop_directive(
                        tokens): # most complex first
                    parse_result = tree.grammar.loop_annotation.parseString(
                        stmt, parseAll=True)
                    curr_offload_region = parse_result[0]
                    curr_offload_loop = parse_result[0]
                    util.logging.log_debug2(
                        opts.log_prefix, "_parse_fortran_code.append_",
                        "found {} in statement '{}'".format(
                            "loop offloading directive", stmt))
                elif util.parsing.is_fortran_offload_region_directive(tokens):
                    parse_result = tree.grammar.parallel_region_start.parseString(
                        stmt, parseAll=True)
                    curr_offload_region = parse_result[0]
                    util.logging.log_debug2(
                        opts.log_prefix, "_parse_fortran_code.append_",
                        "found {} in statement '{}'".format(
                            "begin of offloaded region", stmt))
                elif util.parsing.is_fortran_offload_loop_directive(tokens):
                    parse_result = tree.grammar.loop_annotation.parseString(
                        stmt, parseAll=True)
                    curr_offload_loop = parse_result[0]
                    util.logging.log_debug2(
                        opts.log_prefix, "_parse_fortran_code.append_",
                        "found {} in statement '{}'".format(
                            "loop directive", stmt))
                else:
                    warn_("directive", e)
            except Exception as e:
                error_("directive", e)
                pass
        # do/while
        elif util.parsing.is_do_while(tokens):
            try:
                parse_result = tree.grammar.fortran_do_while.parseString(
                    stmt_no_comment, parseAll=True)
                descend_(tree.TTDoWhile(stmt, 0,
                                        parse_result.asList() + [[]]),
                         "do-while loop")
            except Exception as e:
                error_("do-while loop", e)
        elif util.parsing.is_do(tokens):
            try:
                parse_result = tree.grammar.fortran_do.parseString(
                    stmt_no_comment, parseAll=True)
                do_loop_tokens = [curr_offload_loop
                                 ] + parse_result.asList() + [[]]
                do_loop = tree.TTDo(stmt, 0, do_loop_tokens)
                if curr_offload_region != None:
                    descend_(tree.TTLoopNest(stmt_no_comment,"", [curr_offload_region, [do_loop]]),\
                            "offloaded do loop")
                    do_loop.parent = curr
                    curr = do_loop
                    curr_offload_region = None
                else:
                    descend_(do_loop, "do loop")
                curr_offload_loop = None
            except Exception as e:
                error_("do loop", e)
        # if-then-else
        elif util.parsing.is_if_then(tokens):
            try:
                descend_(tree.TTIfElseBlock(), "if block", inc_level=False)
                parse_result = tree.grammar.fortran_if_else_if.parseString(
                    stmt_no_comment, parseAll=True)
                descend_(
                    tree.TTIfElseIf(stmt_no_comment, 0,
                                    parse_result.asList() + [[]]), "if branch")
            except Exception as e:
                error_("if", e)
        elif util.parsing.is_else_if_then(tokens):
            assert type(curr) is tree.TTIfElseIf
            ascend_("if")
            try:
                parse_result = tree.grammar.fortran_if_else_if.parseString(
                    stmt_no_comment, parseAll=True)
                descend_(
                    tree.TTIfElseIf(stmt_no_comment, 0,
                                    parse_result.asList() + [[]]),
                    "else-if branch")
            except Exception as e:
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
                    tree.TTSelectCase(stmt_no_comment, 0,
                                      parse_result.asList() + [[]]),
                    "select-case")
            except Exception as e:
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
            except Exception as e:
                error_("else-if", e)
        elif util.parsing.is_case_default(tokens):
            if type(curr) is tree.TTCase:
                ascend_("case")
            descend_(tree.TTCase(stmt_no_comment, 0, [[]]), "case default")
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
                parse_result = tree.grammar.fortran_assignment.parseString(
                    stmt_no_comment, parseAll=True)
                append_(parse_result[0], "assignment")
            except Exception as e:
                error_("assignment", e)
        elif util.parsing.is_subroutine_call(tokens):
            try:
                parse_result = tree.grammar.fortran_subroutine_call.parseString(
                    stmt_no_comment, parseAll=True)
                append_(parse_result[0], "subroutine call")
            except Exception as e:
                error_("subroutine call", e)
        else:
            error_("unknown and not ignored")

    return ttree


# API
@util.logging.log_entry_and_exit(opts.log_prefix)
def parse_declaration(fortran_statement):
    orig_tokens = util.parsing.tokenize(fortran_statement.lower(),
                                        padded_size=10)
    tokens = orig_tokens
    util.logging.log_debug2(opts.log_prefix, "parse_declaration",
                            "tokens=" + str(tokens))

    idx_last_consumed_token = None
    # handle datatype
    datatype = tokens[0]
    kind = None
    DOUBLE_COLON = "::"
    if tokens[0] == "type":
        # ex: type ( dim3 )
        kind = tokens[2]
        idx_last_consumed_token = 3
    elif tokens[0:1 + 1] == ["double", "precision"]:
        datatype = " ".join(tokens[0:1 + 1])
        idx_last_consumed_token = 1
    elif tokens[1] == "*":
        # ex: integer * 4
        # ex: integer * ( 4*2 )
        kind_tokens = util.parsing.next_tokens_till_open_bracket_is_closed(
            tokens[2:], open_brackets=0)
        kind = "".join(kind_tokens)
        idx_last_consumed_token = 2 + len(kind_tokens) - 1
    elif tokens[1] == "(" and tokens[2] in ["kind","len"] and\
         tokens[3] in ["=","("]:
        # ex: integer ( kind = 4 )
        # ex: integer ( kind ( 4 ) )
        kind_tokens = util.parsing.next_tokens_till_open_bracket_is_closed(
            tokens[3:], open_brackets=1)
        kind = "".join(kind_tokens[:-1])
        idx_last_consumed_token = 3 + len(kind_tokens) - 1
    elif tokens[1] == "(":
        # ex: integer ( 4 )
        # ex: integer ( 4*2 )
        kind_tokens = util.parsing.next_tokens_till_open_bracket_is_closed(
            tokens[2:], open_brackets=1)
        kind = "".join(kind_tokens[:-1])
        idx_last_consumed_token = 2 + len(kind_tokens) - 1
    elif tokens[1] in [",", DOUBLE_COLON] or tokens[1].isidentifier():
        # ex: integer ,
        # ex: integer ::
        # ex: integer a
        idx_last_consumed_token = 0
    else:
        util.logging.log_error(
            opts.log_prefix, "parse_declaration",
            "could not parse datatype in variable declaration statement '{}'"
            .format(fortran_statement))
        sys.exit(2)

    # handle qualifiers
    tokens = tokens[idx_last_consumed_token + 1:] # remove type part tokens
    idx_last_consumed_token = None
    if tokens[0] == "," and DOUBLE_COLON in tokens:
        qualifiers = util.parsing.get_highest_level_arguments(tokens)
        idx_last_consumed_token = tokens.index(DOUBLE_COLON)
    elif tokens[0] == DOUBLE_COLON:
        idx_last_consumed_token = 0
    elif tokens[0] == ",":
        util.logging.log_error(
            opts.log_prefix, "parse_declaration",
            "could not parse qualifier list in variable declaration statement '{}'"
            .format(fortran_statement))
        sys.exit(2)
    qualifiers_raw = util.parsing.get_highest_level_arguments(tokens)

    # handle variables list
    if idx_last_consumed_token != None:
        tokens = tokens[idx_last_consumed_token
                        + 1:] # remove qualifier list tokens
    variables_raw = util.parsing.get_highest_level_arguments(tokens)

    util.logging.log_debug2(
        opts.log_prefix, "parse_declaration",
        "type=" + str(datatype) + ",kind=" + str(kind) + ",qualifiers="
        + str(qualifiers_raw) + "," + "variables=" + str(variables_raw))
    # construct declaration tree node
    qualifiers = []
    for qualifier in qualifiers_raw:
        if qualifier.startswith("dimension"):
            try:
                qualifiers.append(
                    tree.grammar.dimension_qualifier.parseString(
                        qualifier, parseAll=True)[0])
            except:
                util.logging.log_error(
                    opts.log_prefix, "parse_declaration",
                    "could not parse dimension qualifier in variable declaration statement '{}'"
                    .format(fortran_statement))
                sys.exit(2)
        elif qualifier.startswith("intent"):
            # ex: intent ( inout )
            intent_tokens = util.parsing.tokenize(qualifier)
            if len(intent_tokens) != 4 or intent_tokens[2] not in ["out","inout","in"]:
                util.logging.log_error(
                    opts.log_prefix, "parse_declaration",
                    "could not parse intent qualifier in variable declaration statement '{}'"
                    .format(fortran_statement))
                sys.exit(2)
            qualifiers.append(qualifier)
        else:
            qualifiers.append(qualifier)
    variables = []
    for var in variables_raw:
        try:
            variables.append(
                tree.grammar.declared_var.parseString(var,
                                                           parseAll=True)[0])
        except:
            util.logging.log_error(
                opts.log_prefix, "parse_declaration",
                "could not parse variable declaration '{}' in statement '{}'"
                .format(var, fortran_statement))
            sys.exit(2)
    declaration_tokens = [datatype, kind, qualifiers, variables]
    result = tree.TTDeclaration(fortran_statement, 0, declaration_tokens)
    return result

@util.logging.log_entry_and_exit(opts.log_prefix)
def create_index_records_from_declaration(ttdeclaration):
    """
    Per declared variable in the declaration, creates
    a context dictionary that can be easily piped
    to the (HIP) kernel code generation
    and the Fortran-C interface code generation.
    """
    context = []
    has_dimension = ttdeclaration.has_dimension()
    for ttdeclaredvariable in ttdeclaration._rhs:
        var_name = ttdeclaredvariable.name().lower()
        ivar = {}
        # basic
        f_type = tree.make_f_str(ttdeclaration.type)
        kind = tree.make_f_str(ttdeclaration.kind)
        bpe = conv.num_bytes(f_type, kind, default=None)
        # TODO substitute parameters in scope and try again to compute bpe if entry is None
        ivar["name"] = var_name
        ivar["f_type"] = f_type
        ivar["kind"] = kind
        ivar["bytes_per_element"] = bpe
        ivar["c_type"] = conv.convert_to_c_type(f_type, kind, "TODO unknown")
        ivar["f_interface_type"] = ivar["f_type"]
        ivar["f_interface_qualifiers"] = ["value"
                                         ] # assume pass by value by default
        # TODO pack into single variable
        ivar["qualifiers"] = ttdeclaration.get_string_qualifiers()
        # ACC/OMP
        ivar["declare_on_target"] = False
        # arrays
        ivar["rank"] = 0
        dimension_qualifier = tree.find_all(ttdeclaration.qualifiers,
                                            tree.TTDimensionQualifier)
        # TODO(refactor) some of the information below will become unnecessary
        if ttdeclaredvariable.has_bounds() or len(dimension_qualifier):
            ivar["f_interface_type"] = "type(c_ptr)"
            if len(dimension_qualifier):
                ttbounds = dimension_qualifier[0]._bounds
            else:
                ttbounds = ttdeclaredvariable.get_bounds()
            rank = ttbounds.rank()
            ivar["rank"] = rank
            ivar["unspecified_bounds"] = ttbounds.has_unspecified_bounds()
            if ivar["unspecified_bounds"]: # TODO: return a mix of unspecified bounds and specified bounds in the future
                ivar["lbounds"] = [
                    "{0}_lb{1}".format(var_name, i)
                    for i in range(1, rank + 1)
                ]
                ivar["counts"] = [
                    "{0}_n{1}".format(var_name, i) for i in range(1, rank + 1)
                ]
                ivar[
                    "index_macro_with_placeholders"] = ttbounds.index_macro_c_str(
                        var_name, use_place_holders=True)
                ivar["index_macro"] = ivar["index_macro_with_placeholders"]
            else:
                ivar["lbounds"] = ttbounds.specified_lower_bounds()
                ivar["counts"] = ttbounds.specified_counts()
                ivar[
                    "index_macro_with_placeholders"] = ttbounds.index_macro_c_str(
                        var_name, use_place_holders=True)
                ivar["index_macro"] = ttbounds.index_macro_c_str(
                    var_name, use_place_holders=False)
            ivar["total_count"] = "*".join(ivar["counts"])
            ivar["total_bytes"] = None if bpe is None else bpe + "*(" + ivar[
                "total_count"] + ")"
        # handle parameters
        ivar["value"] = None
        if "parameter" in ivar["qualifiers"]:
            if ttdeclaredvariable.rhs_is_number():
                ivar["value"] = ttdeclaredvariable.rhs_c_str()
            else:
                #ivar["value"] = ttdeclaredvariable.rhs_c_str()
                # TODO
                pass
        context.append(ivar)

    return context


def change_kind(ivar, kind):
    f_type = ivar["f_type"]
    bpe = conv.num_bytes(f_type, kind, default=None)
    ivar["kind"] = kind
    ivar["bytes_per_element"] = bpe
    ivar["c_type"] = conv.convert_to_c_type(f_type, kind, "TODO unknown")
    if ivar["rank"] == 0:
        ivar["f_interface_type"] = ivar["c_type"]
    #
    ivar["total_bytes"] = None if bpe is None else bpe + "*(" + ivar[
        "total_count"] + ")"


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


def parse_loop_kernel(fortran_statements, scope=None):
    """:return: C snippet equivalent to original Fortran code.
    """
    ttloopkernel = _parse_fortran_code(fortran_statements).body[0]

    ttloopkernel.scope = scope
    return ttloopkernel


def parse_procedure_body(fortran_statements, scope=None, result_name=""):
    """Parse a function/subroutine body.
    """
    parse_result = _parse_fortran_code(fortran_statements)
    ttprocedurebody = tree.TTProcedureBody("", 0, [parse_result.body])

    ttprocedurebody.scope = scope
    ttprocedurebody.result_name = result_name
    return ttprocedurebody
