# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os, sys, traceback
import subprocess
import copy
import argparse
import itertools
import hashlib
from collections import Iterable # < py38
import importlib

import re
import pyparsing

# local includes
from gpufort import util
from gpufort import linemapper
from gpufort import translator
from gpufort import indexer
from . import tree
from . import opts


# Pyparsing actions that create scanner tree (ST)
@util.logging.log_entry_and_exit(opts.log_prefix)
def _parse_file(linemaps, index):
    """Generates a tree representation of the Fortran constructs
    in a Fortran file that is called *scanner tree*.
    :param list linemaps: List of linemaps created by GPUFORT's linemapper
                          component.
    :param list index: Index records created by GPUFORT's indexer
                       component.
    """

    translation_enabled = opts.translation_enabled_by_default

    current_node = tree.STRoot()
    do_loop_ctr = 0
    keep_recording = False
    fortran_file_path = linemaps[0]["file"]
    current_file = str(fortran_file_path)
    current_linemap = None
    directive_no = 0

    def log_detection_(kind):
        nonlocal current_node
        nonlocal current_linemap
        util.logging.log_debug2(opts.log_prefix,"parse_file","[current-node={}:{}] found {} in line {}: '{}'".format(\
                current_node.kind,current_node.name,kind,current_linemap["lineno"],current_linemap["lines"][0].rstrip("\n")))

    def append_if_not_recording_(new):
        nonlocal current_node
        nonlocal keep_recording
        if not keep_recording:
            new.parent = current_node
            current_node.append(new)

    def descend_(new):
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        parent_node_id = current_node.kind
        parent_node_name = current_node.name

        new.parent = current_node
        current_node.append(new)
        current_node = new

        current_node_id = current_node.kind
        if current_node.name != None:
            current_node_id += " '" + current_node.name + "'"
        if parent_node_name != None:
            parent_node_id += ":" + parent_node_name

        util.logging.log_debug(opts.log_prefix,"parse_file","[current-node={0}] enter {1} in line {2}: '{3}'".format(\
          parent_node_id,current_node_id,current_linemap["lineno"],current_linemap["lines"][0].rstrip("\n")))

    def ascend_():
        nonlocal current_node
        nonlocal current_file
        nonlocal current_linemap
        nonlocal current_statement_no
        assert not current_node.parent is None, "In file {}: parent of {} is none".format(
            current_file, type(current_node))
        current_node_id = current_node.kind
        if current_node.name != None:
            current_node_id += " '" + current_node.name + "'"
        parent_node_id = current_node.parent.kind
        if current_node.parent.name != None:
            parent_node_id += ":" + current_node.parent.name

        util.logging.log_debug(opts.log_prefix,"parse_file","[current-node={0}] leave {1} in line {2}: '{3}'".format(\
          parent_node_id,current_node_id,current_linemap["lineno"],current_linemap["lines"][0].rstrip("\n")))
        current_node = current_node.parent

    # parse actions
    def Module_visit(tokens):
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        log_detection_("module")
        new = tree.STModule(tokens[0], current_linemap, current_statement_no)
        new.ignore_in_s2s_translation = not translation_enabled
        descend_(new)

    def Program_visit(tokens):
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        log_detection_("program")
        new = tree.STProgram(tokens[0], current_linemap, current_statement_no)
        new.ignore_in_s2s_translation = not translation_enabled
        descend_(new)

    def Function_visit(tokens):
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal keep_recording
        nonlocal index
        log_detection_("function")
        new = tree.STProcedure(tokens[1],current_node.tag(),"function",\
            current_linemap,current_statement_no,index)
        new.ignore_in_s2s_translation = not translation_enabled
        keep_recording = new.keep_recording()
        descend_(new)

    def Subroutine_visit(tokens):
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal keep_recording
        nonlocal index
        log_detection_("subroutine")
        new = tree.STProcedure(tokens[1],current_node.tag(),"subroutine",\
            current_linemap,current_statement_no,index)
        new.ignore_in_s2s_translation = not translation_enabled
        keep_recording = new.keep_recording()
        descend_(new)

    def End():
        nonlocal current_node
        nonlocal current_linemap
        nonlocal keep_recording
        nonlocal index
        log_detection_("end of module/program/function/subroutine")
        assert type(current_node) in [
            tree.STModule, tree.STProgram, tree.STProcedure
        ], "In file {}: line {}: type is {}".format(current_file,
                                                    current_statement_no,
                                                    type(current_node))
        if type(
                current_node
        ) is tree.STProcedure and current_node.must_be_available_on_device():
            current_node.add_linemap(current_linemap)
            current_node._last_statement_index = current_statement_no
            current_node.complete_init(index)
            keep_recording = False
        if not keep_recording:
            new = tree.STEnd(current_linemap, current_statement_no)
            append_if_not_recording_(new)
        ascend_()

    def Return():
        nonlocal current_node
        nonlocal current_linemap
        nonlocal keep_recording
        log_detection_("return statement")
        if not keep_recording and isinstance(
                current_node, (tree.STProcedure, tree.STProgram)):
            new = tree.STReturn(current_linemap, current_statement_no)
            append_if_not_recording_(new)

    def in_kernels_acc_region_and_not_recording():
        nonlocal current_node
        nonlocal keep_recording
        return not keep_recording and\
            (type(current_node) is tree.acc.STAccDirective) and\
            (current_node.is_kernels_directive())

    def DoLoop_visit():
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal do_loop_ctr
        nonlocal keep_recording
        log_detection_("do loop")
        if in_kernels_acc_region_and_not_recording():
            new = tree.acc.STAccLoopNest(current_linemap, current_statement_no)
            new.ignore_in_s2s_translation = not translation_enabled
            new._do_loop_ctr_memorised = do_loop_ctr
            descend_(new)
            keep_recording = True
        do_loop_ctr += 1

    def DoLoop_leave():
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        nonlocal do_loop_ctr
        nonlocal keep_recording
        nonlocal index
        log_detection_("end of do loop")
        do_loop_ctr -= 1
        if isinstance(current_node, tree.STLoopNest):
            if keep_recording and current_node._do_loop_ctr_memorised == do_loop_ctr:
                current_node.add_linemap(current_linemap)
                current_node._last_statement_index = current_statement_no
                current_node.complete_init(index)
                ascend_()
                keep_recording = False

    def Declaration():
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        log_detection_("declaration")
        new = tree.STDeclaration(current_linemap, current_statement_no)
        new.ignore_in_s2s_translation = not translation_enabled
        append_if_not_recording_(new)

    def Attributes(tokens):
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        log_detection_("attributes statement")
        new = tree.STAttributes(current_linemap, current_statement_no)
        new.ignore_in_s2s_translation = not translation_enabled
        current_node.append(new)

    def UseStatement(tokens):
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        log_detection_("use statement")
        new = tree.STUseStatement(current_linemap, current_statement_no)
        new.ignore_in_s2s_translation = not translation_enabled
        new.name = translator.tree.make_f_str(
            tokens[1]) # just get the name, ignore specific includes
        append_if_not_recording_(new)

    def PlaceHolder():
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        log_detection_("placeholder")
        new = tree.STPlaceHolder(current_linemap, current_statement_no)
        new.ignore_in_s2s_translation = not translation_enabled
        append_if_not_recording_(new)

    def Contains():
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        log_detection_("contains")
        new = tree.STContains(current_linemap, current_statement_no)
        new.ignore_in_s2s_translation = not translation_enabled
        append_if_not_recording_(new)

    def NonZeroCheck(tokens):
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        log_detection_("non-zero check")
        new = tree.STNonZeroCheck(current_linemap, current_statement_no)
        new.ignore_in_s2s_translation = not translation_enabled
        append_if_not_recording_(new)

    def Allocated(tokens):
        nonlocal current_node
        nonlocal translation_enabled
        nonlocal current_linemap
        nonlocal current_statement_no
        log_detection_("allocated statement")
        new = tree.STAllocated(current_linemap, current_statement_no)
        new.ignore_in_s2s_translation = not translation_enabled
        append_if_not_recording_(new)

    def Allocate(tokens):
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        log_detection_("allocate statement")
        new = tree.STAllocate(current_linemap, current_statement_no)
        new.ignore_in_s2s_translation = not translation_enabled
        append_if_not_recording_(new)

    def Deallocate(tokens):
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        log_detection_("deallocate statement")
        # TODO filter variable, replace with hipFree
        new = tree.STDeallocate(current_linemap, current_statement_no)
        new.ignore_in_s2s_translation = not translation_enabled
        append_if_not_recording_(new)

    def Memcpy(tokens):
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        log_detection_("memcpy")
        new = tree.STMemcpy(current_linemap, current_statement_no)
        new.ignore_in_s2s_translation = not translation_enabled
        append_if_not_recording_(new)

    def CufLibCall(tokens):
        #TODO scan for cudaMemcpy calls
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        nonlocal keep_recording
        log_detection_("CUDA API call")
        cudaApi, args = tokens
        if not type(current_node) in [cuf.STCufLibCall, cuf.STCufKernelCall]:
            new = tree.cuf.STCufLibCall(current_linemap, current_statement_no)
            new.ignore_in_s2s_translation = not translation_enabled
            new.cudaApi = cudaApi
            #print("finishes_on_first_line={}".format(finishes_on_first_line))
            assert type(new.parent) in [
                tree.STModule, tree.STProcedure, tree.STProgram
            ], type(new.parent)
            append_if_not_recording_(new)

    def CufKernelCall(tokens):
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        nonlocal keep_recording
        log_detection_("CUDA kernel call")
        kernel_name, kernel_launch_args, args = tokens
        assert type(current_node) in [
            tree.STModule, tree.STProcedure, tree.STProgram
        ], "type is: " + str(type(current_node))
        new = tree.cuf.STCufKernelCall(current_linemap, current_statement_no)
        new.ignore_in_s2s_translation = not translation_enabled
        append_if_not_recording_(new)

    def AccDirective():
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        nonlocal keep_recording
        nonlocal do_loop_ctr
        nonlocal directive_no
        log_detection_("OpenACC directive")
        new = tree.acc.STAccDirective(current_linemap, current_statement_no,
                                      directive_no)
        new.ignore_in_s2s_translation = not translation_enabled
        directive_no += 1
        # if end directive ascend
        if new.is_end_directive() and type(current_node) is tree.acc.STAccDirective and\
           current_node.is_kernels_directive():
            ascend_()
        # descend in constructs or new node
        elif new.is_parallel_loop_directive() or new.is_kernels_loop_directive() or\
             (not new.is_end_directive() and new.is_parallel_directive()):
            new = tree.acc.STAccLoopNest(current_linemap, current_statement_no,
                                         directive_no)
            new.kind = "acc-compute-construct"
            new.ignore_in_s2s_translation = not translation_enabled
            new._do_loop_ctr_memorised = do_loop_ctr
            descend_(new) # descend also appends
            keep_recording = True
        elif not new.is_end_directive() and new.is_kernels_directive():
            descend_(new) # descend also appends
            #print("descending into kernels or parallel construct")
        else:
            # append new directive
            append_if_not_recording_(new)

    def CufLoopNest():
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        nonlocal do_loop_ctr
        nonlocal keep_recording
        nonlocal directive_no
        log_detection_("CUDA Fortran loop kernel directive")
        new = tree.cuf.STCufLoopNest(current_linemap, current_statement_no,
                                     directive_no)
        new.kind = "cuf-kernel-do"
        new.ignore_in_s2s_translation = not translation_enabled
        new._do_loop_ctr_memorised = do_loop_ctr
        directive_no += 1
        descend_(new)
        keep_recording = True

    def Assignment(tokens):
        nonlocal current_node
        nonlocal translation_enabled
        nonlocal current_linemap
        nonlocal current_statement_no
        nonlocal current_statement
        log_detection_("assignment")
        if in_kernels_acc_region_and_not_recording():
            parse_result = translator.assignment_begin.parseString(
                current_statement["body"])
            lvalue = translator.find_first(parse_result, translator.TTLValue)
            if not lvalue is None and lvalue.has_matrix_range_args():
                new = tree.acc.STAccLoopNest(current_linemap,
                                             current_statement_no)
                new.ignore_in_s2s_translation = not translation_enabled
                append_if_not_recording_(new)

    def GpufortControl():
        nonlocal current_statement
        nonlocal translation_enabled
        log_detection_("gpufortran control statement")
        if "on" in current_statement["body"]:
            translation_enabled = True
        elif "off" in current_statement["body"]:
            translation_enabled = False

    # TODO completely remove / comment out !$acc end kernels
    tree.grammar.module_start.setParseAction(Module_visit)
    tree.grammar.program_start.setParseAction(Program_visit)
    tree.grammar.function_start.setParseAction(Function_visit)
    tree.grammar.subroutine_start.setParseAction(Subroutine_visit)

    tree.grammar.use.setParseAction(UseStatement)

    datatype_reg = pyparsing.Regex(
        r"\s*\b(type\s*\(\s*\w+\s*\)|character|integer|logical|real|complex|double\s+precision)\b"
    )
    datatype_reg.setParseAction(Declaration)

    tree.grammar.attributes.setParseAction(Attributes)
    tree.grammar.ALLOCATED.setParseAction(Allocated)
    tree.grammar.ALLOCATE.setParseAction(Allocate)
    tree.grammar.DEALLOCATE.setParseAction(Deallocate)
    tree.grammar.memcpy.setParseAction(Memcpy)
    #pointer_assignment.setParseAction(pointer_assignment)
    tree.grammar.non_zero_check.setParseAction(NonZeroCheck)

    # CUDA Fortran
    tree.grammar.cuda_lib_call.setParseAction(CufLibCall)
    tree.grammar.cuf_kernel_call.setParseAction(CufKernelCall)

    # OpenACC
    tree.grammar.ACC_START.setParseAction(AccDirective)
    tree.grammar.assignment_begin.setParseAction(Assignment)

    # GPUFORT control
    tree.grammar.gpufort_control.setParseAction(GpufortControl)

    current_file = str(fortran_file_path)
    current_node.children.clear()

    def scan_string(expression_name, expression):
        """
        These expressions might be hidden behind a single-line if.
        """
        nonlocal current_linemap
        nonlocal current_statement_no
        nonlocal current_statement

        matched = len(expression.searchString(current_statement["body"], 1))
        if matched:
            util.logging.log_debug3(
                opts.log_prefix, "parse_file.scanString",
                "found expression '{}' in line {}: '{}'".format(
                    expression_name, current_linemap["lineno"],
                    current_linemap["lines"][0].rstrip()))
        else:
            util.logging.log_debug4(
                opts.log_prefix, "parse_file.scanString",
                "did not find expression '{}' in line {}: '{}'".format(
                    expression_name, current_linemap["lineno"],
                    current_linemap["lines"][0].rstrip()))
        return matched

    def try_to_parse_string(expression_name, expression, parseAll=False):
        """
        These expressions might never be hidden behind a single-line if or might
        never be an argument of another calls.
        No need to check if comments are in the line as the first
        words of the line must match the expression.
        """
        nonlocal current_linemap
        nonlocal current_statement_no
        nonlocal current_statement_stripped_no_comments

        try:
            expression.parseString(current_statement_stripped_no_comments,
                                   parseAll)
            util.logging.log_debug3(
                opts.log_prefix, "parse_file.try_to_parse_string",
                "found expression '{}' in line {}: '{}'".format(
                    expression_name, current_linemap["lineno"],
                    current_linemap["lines"][0].rstrip()))
            return True
        except pyparsing.ParseBaseException as e:
            util.logging.log_debug4(
                opts.log_prefix, "parse_file.try_to_parse_string",
                "did not find expression '{}' in line '{}'".format(
                    expression_name, current_linemap["lines"][0]))
            util.logging.log_debug5(opts.log_prefix,
                                    "parse_file.try_to_parse_string", str(e))
            return False

    def is_end_statement_(tokens, kind):
        result = tokens[0] == "end" + kind
        if not result and len(tokens):
            result = tokens[0] == "end" and tokens[1] == kind
        return result

    # parser loop
    for current_linemap in linemaps:
        condition1 = current_linemap["is_active"]
        condition2 = len(current_linemap["included_linemaps"]
                        ) or not current_linemap["is_preprocessor_directive"]
        if condition1 and condition2:
            for current_statement_no, current_statement in enumerate(
                    current_linemap["statements"]):
                util.logging.log_debug4(opts.log_prefix,"parse_file","parsing statement '{}' associated with lines [{},{}]".format(current_statement["body"].rstrip(),\
                    current_linemap["lineno"],current_linemap["lineno"]+len(current_linemap["lines"])-1))

                current_tokens = util.parsing.tokenize(
                    current_statement["body"].lower(), padded_size=6)
                current_statement_stripped = " ".join(current_tokens)
                current_statement_stripped_no_comments = current_statement_stripped.split(
                    "!")[0]
                if len(current_tokens):
                    # constructs
                    if current_tokens[0] == "end":
                        for kind in [
                                "program", "module", "subroutine", "function"
                        ]: # ignore types/interfaces here
                            if is_end_statement_(current_tokens, kind):
                                End()
                        if is_end_statement_(current_tokens, "do"):
                            DoLoop_leave()
                    if util.parsing.is_do(current_tokens):
                        DoLoop_visit()
                    # single-statements
                    if not keep_recording:
                        if "acc" in opts.source_dialects:
                            for comment_char in "!*c":
                                if current_tokens[0:2] == [
                                        comment_char + "$", "acc"
                                ]:
                                    AccDirective()
                                elif current_tokens[0:2] == [
                                        comment_char + "$", "gpufort"
                                ]:
                                    GpufortControl()
                        if "cuf" in opts.source_dialects:
                            if "attributes" in current_tokens:
                                try_to_parse_string("attributes",
                                                    tree.grammar.attributes)
                            if "cu" in current_statement_stripped_no_comments:
                                scan_string("cuda_lib_call",
                                            tree.grammar.cuda_lib_call)
                            if "<<<" in current_tokens:
                                try_to_parse_string(
                                    "cuf_kernel_call",
                                    tree.grammar.cuf_kernel_call)
                            for comment_char in "!*c":
                                if current_tokens[0:2] == [
                                        comment_char + "$", "cuf"
                                ]:
                                    CufLoopNest()
                        if "=" in current_tokens:
                            if not try_to_parse_string("memcpy",
                                                       tree.grammar.memcpy,
                                                       parseAll=True):
                                try_to_parse_string(
                                    "assignment",
                                    tree.grammar.assignment_begin)
                                scan_string("non_zero_check",
                                            tree.grammar.non_zero_check)
                        if "allocated" in current_tokens:
                            scan_string("allocated", tree.grammar.ALLOCATED)
                        if "deallocate" in current_tokens:
                            try_to_parse_string("deallocate",
                                                tree.grammar.DEALLOCATE)
                        if "allocate" in current_tokens:
                            try_to_parse_string("allocate",
                                                tree.grammar.ALLOCATE)
                        if "function" in current_tokens:
                            try_to_parse_string("function",
                                                tree.grammar.function_start)
                        if "subroutine" in current_tokens:
                            try_to_parse_string("subroutine",
                                                tree.grammar.subroutine_start)
                        #
                        if current_tokens[0] == "use":
                            try_to_parse_string("use", tree.grammar.use)
                        elif current_tokens[0] == "implicit":
                            PlaceHolder()
                        elif current_tokens[0] == "contains":
                            Contains()
                        elif current_tokens[0] == "module":
                            try_to_parse_string("module",
                                                tree.grammar.module_start)
                        elif current_tokens[0:2] == ["end", "type"]:
                            PlaceHolder()
                        elif current_tokens[0] == "program":
                            try_to_parse_string("program",
                                                tree.grammar.program_start)
                        elif current_tokens[0] == "return":
                            Return()
                        elif current_tokens[0] in [
                                "character", "integer", "logical", "real",
                                "complex", "double"
                        ]:
                            try_to_parse_string("declaration", datatype_reg)
                            break
                        elif current_tokens[0] == "type" and current_tokens[
                                1] == "(":
                            try_to_parse_string("declaration", datatype_reg)
                    else:
                        current_node.add_linemap(current_linemap)
                        current_node._last_statement_index = current_statement_no

    assert type(current_node) is tree.STRoot
    return current_node


def parse_file(**kwargs):
    r"""Create scanner tree from file content.
    :param \*\*kwargs: See below.
    :return: a triple of scanner tree root node, (updated) index, and linemaps

    :Keyword Arguments:
    
    * *file_path* (``str``):
        Path to the file that should be parsed.
    * *file_content* (``str``):
        Content of the file that should be parsed.
    * *file_linemaps* (``list``):
        Linemaps of the file that should be parsed;
        see GPUFORT's linemapper component.
    * *file_is_indexed* (``bool``):
        Index already contains entries for the main file.
    * *preproc_options* (``str``): Options to pass to the C preprocessor
        of the linemapper component.
    * *other_files_paths* (``list(str)``):
        Paths to other files that contain module files that
        contain definitions required for parsing the main file.
        NOTE: It is assumed that these files have not been indexed yet.
    * *other_files_contents* (``list(str)``):
        Content that contain module files that
        contain definitions required for parsing the main file.
        NOTE: It is assumed that these files have not been indexed yet.
    * *index* (``list``):
        Index records created via GPUFORT's indexer component.
    """
    # Determine if there is an index arg and
    # what has already been indexed
    index, have_index = util.kwargs.get_value("index", [], **kwargs)
    file_is_indexed, _ = util.kwargs.get_value("file_is_indexed", have_index,
                                               **kwargs)
    # handle main file
    file_content, have_file_content = util.kwargs.get_value(
        "file_content", None, **kwargs)
    linemaps, have_linemaps = util.kwargs.get_value("file_linemaps", None,
                                                    **kwargs)
    file_path, have_file_path = util.kwargs.get_value("file_path", "<unknown>",
                                                      **kwargs)
    if not have_file_content and not have_linemaps:
        if not have_file_path:
            raise ValueError(
                "Missing keyword argument: At least one of `file_path`, `file_content`, `linemaps` must be specified."
            )
        with open(file_path, "r") as infile:
            file_content = infile.read()
    preproc_options, _ = util.kwargs.get_value("preproc_options", "", **kwargs)
    if not have_linemaps:
        linemaps = linemapper.read_lines(
            file_content.splitlines(keepends=True), preproc_options, file_path)
    if not file_is_indexed:
        indexer.update_index_from_linemaps(linemaps, index)

    # handles dependencies
    other_files_contents, have_other_files_contents = util.kwargs.get_value(
        "other_files_contents", [], **kwargs)
    other_files_paths, have_other_files_paths = util.kwargs.get_value(
        "other_files_paths", [], **kwargs)
    if have_other_files_paths:
        for path in other_files_paths:
            with open(path, "r") as infile:
                other_files_contents.append(infile.read())
    for content in other_files_contents:
        indexer.update_index_from_snippet(index, content, preproc_options)
    return _parse_file(linemaps, index), index, linemaps


__all__ = ["parse_file"]
