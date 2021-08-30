# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
#!/usr/bin/env python3
import os, sys, traceback
import subprocess
import copy
import argparse
import itertools
import hashlib
from collections import Iterable # < py38
import importlib
import logging

import re

# local includes
import addtoplevelpath
import translator.translator as translator
import indexer.scoper as scoper
import utils.pyparsingutils
#import scanner.normalizer as normalizer

SCANNER_ERROR_CODE = 1000
#SUPPORTED_DESTINATION_DIALECTS = ["omp","hip-gpufort-rt","hip-gcc-rt","hip-hpe-rt","hip"]
SUPPORTED_DESTINATION_DIALECTS = []
RUNTIME_MODULE_NAMES = {}

# dirty hack that allows us to load independent versions of the grammar module
#from grammar import *
CASELESS    = True
GRAMMAR_DIR = os.path.join(os.path.dirname(__file__),"../grammar")
exec(open("{0}/grammar.py".format(GRAMMAR_DIR)).read())
scanner_dir = os.path.dirname(__file__)
exec(open("{0}/scanner_options.py.in".format(scanner_dir)).read())
exec(open("{0}/scanner_tree.py.in".format(scanner_dir)).read())
exec(open("{0}/openacc/scanner_tree_acc.py.in".format(scanner_dir)).read())
exec(open("{0}/cudafortran/scanner_tree_cuf.py.in".format(scanner_dir)).read())

def check_destination_dialect(destination_dialect):
    if destination_dialect in SUPPORTED_DESTINATION_DIALECTS:
        return destination_dialect
    else:
        msg = "scanner: destination dialect '{}' is not supported. Must be one of: {}".format(\
                destination_dialect,", ".join(SUPPORTED_DESTINATION_DIALECTS))
        utils.logging.log_error(LOG_PREFIX,"check_destination_dialect",msg)
        sys.exit(SCANNER_ERROR_CODE)

def _intrnl_postprocess_acc(stree):
    """
    Add use statements as well as handles plus their creation and destruction for certain
    math libraries.
    """
    global LOG_PREFIX
    global DESTINATION_DIALECT
    global RUNTIME_MODULE_NAMES
    
    utils.logging.log_enter_function(LOG_PREFIX,"_intrnl_postprocess_acc")
    
    directives = stree.find_all(filter=lambda node: isinstance(node,STAccDirective), recursively=True)
    for directive in directives:
         stnode = directive._parent.find_first(filter=lambda child : not child._ignore_in_s2s_translation and type(child) in [STUseStatement,STDeclaration,STPlaceHolder])
         # add acc use statements
         if not stnode is None:
             indent = stnode.first_line_indent()
             acc_runtime_module_name = RUNTIME_MODULE_NAMES[DESTINATION_DIALECT]
             if acc_runtime_module_name != None and len(acc_runtime_module_name):
                 stnode.add_to_prolog("{0}use {1}\n{0}use iso_c_binding\n".format(indent,acc_runtime_module_name))
        #if type(directive._parent
    utils.logging.log_leave_function(LOG_PREFIX,"_intrnl_postprocess_acc")
    
def _intrnl_postprocess_cuf(stree):
    """
    Add use statements as well as handles plus their creation and destruction for certain
    math libraries.
    """
    global LOG_PREFIX
    global CUBLAS_VERSION 
    utils.logging.log_enter_function(LOG_PREFIX,"_intrnl_postprocess_cuf")
    # cublas_v1 detection
    if CUBLAS_VERSION == 1:
        def has_cublas_call_(child):
            return type(child) is STCudaLibCall and child.has_cublas()
        cuf_cublas_calls = stree.find_all(filter=has_cublas_call_, recursively=True)
        #print(cuf_cublas_calls)
        for call in cuf_cublas_calls:
            begin = call._parent.find_last(filter=lambda child : type(child) in [STUseStatement,STDeclaration])
            indent = self.first_line_indent()
            begin.add_to_epilog("{0}type(c_ptr) :: hipblasHandle = c_null_ptr\n".format(indent))
            #print(begin._linemaps)       
 
            local_cublas_calls = call._parent.find_all(filter=has_cublas_call, recursively=False)
            first = local_cublas_calls[0]
            indent = self.first_line_indent()
            first.add_to_prolog("{0}hipblasCreate(hipblasHandle)\n".format(indent))
            last = local_cublas_calls[-1]
            indent = self.first_line_indent()
            last.add_to_epilog("{0}hipblasDestroy(hipblasHandle)\n".format(indent))
    utils.logging.log_leave_function(LOG_PREFIX,"_intrnl_postprocess_cuf")


# API

# Pyparsing actions that create scanner tree (ST)
def parse_file(linemaps,index,fortran_filepath):
    """
    Generate an object tree (OT). 
    """
    utils.logging.log_enter_function(LOG_PREFIX,"parse_file",
        {"fortran_filepath":fortran_filepath})

    translation_enabled = TRANSLATION_ENABLED_BY_DEFAULT
    
    current_node   = STRoot()
    do_loop_ctr     = 0     
    keep_recording = False
    current_file   = str(fortran_filepath)
    current_linemap = None
    directive_no   = 0

    def log_detection_(kind):
        nonlocal current_node
        nonlocal current_linemap
        utils.logging.log_debug2(LOG_PREFIX,"parse_file","[current-node={}:{}] found {} in line {}: '{}'".format(\
                current_node.kind,current_node.name,kind,current_linemap["lineno"],current_linemap["lines"][0].rstrip("\n")))

    def append_if_not_recording_(new):
        nonlocal current_node
        nonlocal keep_recording 
        if not keep_recording:
            current_node.append(new)
    def descend_(new):
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        current_node.append(new)
        current_node=new
        
        current_node_id = current_node.kind
        if current_node.name != None:
            current_node_id += " '"+current_node.name+"'"
        parent_node_id = current_node._parent.kind
        if current_node._parent.name != None:
            parent_node_id += ":"+current_node._parent.name

        utils.logging.log_debug(LOG_PREFIX,"parse_file","[current-node={0}] enter {1} in line {2}: '{3}'".format(\
          parent_node_id,current_node_id,current_linemap["lineno"],current_linemap["lines"][0].rstrip("\n")))
    def ascend_():
        nonlocal current_node
        nonlocal current_file
        nonlocal current_linemap
        nonlocal current_statement_no
        assert not current_node._parent is None, "In file {}: parent of {} is none".format(current_file,type(current_node))
        
        current_node_id = current_node.kind
        if current_node.name != None:
            current_node_id += " '"+current_node.name+"'"
        parent_node_id = current_node._parent.kind
        if current_node._parent.name != None:
            parent_node_id += ":"+current_node._parent.name
        
        utils.logging.log_debug(LOG_PREFIX,"parse_file","[current-node={0}] leave {1} in line {2}: '{3}'".format(\
          parent_node_id,current_node_id,current_linemap["lineno"],current_linemap["lines"][0].rstrip("\n")))
        current_node = current_node._parent
   
    # parse actions
    def Module_visit(tokens):
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        log_detection_("module")
        new = STModule(tokens[0],current_node,current_linemap,current_statement_no)
        new._ignore_in_s2s_translation = not translation_enabled
        descend_(new)
    def Program_visit(tokens):
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        log_detection_("program")
        new = STProgram(tokens[0],current_node,current_linemap,current_statement_no)
        new._ignore_in_s2s_translation = not translation_enabled
        descend_(new)
    def Function_visit(tokens):
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal keep_recording
        nonlocal index
        log_detection_("function")
        new = STProcedure(tokens[1],"function",\
            current_node,current_linemap,current_statement_no,index)
        new._ignore_in_s2s_translation = not translation_enabled
        keep_recording = new.keep_recording()
        descend_(new)
    def Subroutine_visit(tokens):
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal keep_recording
        nonlocal index
        log_detection_("subroutine")
        new = STProcedure(tokens[1],"subroutine",\
            current_node,current_linemap,current_statement_no,index)
        new._ignore_in_s2s_translation = not translation_enabled
        keep_recording = new.keep_recording()
        descend_(new)
    def End():
        nonlocal current_node
        nonlocal current_linemap
        nonlocal keep_recording
        log_detection_("end of module/program/function/subroutine")
        assert type(current_node) in [STModule,STProgram,STProcedure], "In file {}: line {}: type is {}".format(current_file, current_statement_no, type(current_node))
        if type(current_node) is STProcedure and current_node.must_be_available_on_device():
            current_node.add_linemap(current_linemap)
            current_node._last_statement_index = current_statement_no
            current_node.complete_init()
            keep_recording = False
        if not keep_recording and type(current_node) in [STProcedure,STProgram]:
            new = STEndOrReturn(current_node,current_linemap,current_statement_no)
            append_if_not_recording_(new)
        ascend_()
    def Return():
        nonlocal current_node
        nonlocal current_linemap
        nonlocal keep_recording
        log_detection_("return statement")
        if not keep_recording and type(current_node) in [STProcedure,STProgram]:
            new = STEndOrReturn(current_node,current_linemap,current_statement_no)
            append_if_not_recording_(new)
        ascend_()
    def in_kernels_acc_region_and_not_recording():
        nonlocal current_node
        nonlocal keep_recording
        return not keep_recording and\
            (type(current_node) is STAccDirective) and\
            (current_node.is_kernels_directive())
    def DoLoop_visit():
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal do_loop_ctr
        nonlocal keep_recording
        log_detection_("do loop")
        if in_kernels_acc_region_and_not_recording():
            new = STAccLoopKernel(current_node,current_linemap,current_statement_no)
            new._ignore_in_s2s_translation = not translation_enabled
            new._do_loop_ctr_memorised=do_loop_ctr
            descend_(new) 
            keep_recording = True
        do_loop_ctr += 1
    def DoLoop_leave():
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        nonlocal do_loop_ctr
        nonlocal keep_recording
        log_detection_("end of do loop")
        do_loop_ctr -= 1
        if isinstance(current_node, STLoopKernel):
            if keep_recording and current_node._do_loop_ctr_memorised == do_loop_ctr:
                current_node.add_linemap(current_linemap)
                current_node._last_statement_index = current_statement_no
                current_node.complete_init()
                ascend_()
                keep_recording = False
    def Declaration():
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        log_detection_("declaration")
        new = STDeclaration(current_node,current_linemap,current_statement_no)
        new._ignore_in_s2s_translation = not translation_enabled
        append_if_not_recording_(new)
    def Attributes(tokens):
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        log_detection_("attributes statement")
        new = STAttributes(current_node,current_linemap,current_statement_no)
        new._ignore_in_s2s_translation = not translation_enabled
        current_node.append(new)
    def UseStatement(tokens):
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        log_detection_("use statement")
        new = STUseStatement(current_node,current_linemap,current_statement_no)
        new._ignore_in_s2s_translation = not translation_enabled
        new.name = translator.make_f_str(tokens[1]) # just get the name, ignore specific includes
        append_if_not_recording_(new)
    def PlaceHolder(tokens):
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        log_detection_("placeholder")
        new = STPlaceHolder(current_node,current_linemap,current_statement_no)
        new._ignore_in_s2s_translation = not translation_enabled
        append_if_not_recording_(new)
    def NonZeroCheck(tokens):
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        log_detection_("non-zero check")
        new = STNonZeroCheck(current_node,current_linemap,current_statement_no)
        new._ignore_in_s2s_translation = not translation_enabled
        append_if_not_recording_(new)
    def Allocated(tokens):
        nonlocal current_node
        nonlocal translation_enabled
        nonlocal current_linemap
        nonlocal current_statement_no
        log_detection_("allocated statement")
        new = STAllocated(current_node,current_linemap,current_statement_no)
        new._ignore_in_s2s_translation = not translation_enabled
        append_if_not_recording_(new)
    def Allocate(tokens):
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        log_detection_("allocate statement")
        new = STAllocate(current_node,current_linemap,current_statement_no)
        new._ignore_in_s2s_translation = not translation_enabled
        append_if_not_recording_(new)
    def Deallocate(tokens):
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        log_detection_("deallocate statement")
        # TODO filter variable, replace with hipFree
        new = STDeallocate(current_node,current_linemap,current_statement_no)
        new._ignore_in_s2s_translation = not translation_enabled
        append_if_not_recording_(new)
    def Memcpy(tokens):
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        log_detection_("memcpy")
        new = STMemcpy(current_node,current_linemap,current_statement_no)
        new._ignore_in_s2s_translation = not translation_enabled
        append_if_not_recording_(new)
    def CudaLibCall(tokens):
        #TODO scan for cudaMemcpy calls
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        nonlocal keep_recording
        log_detection_("CUDA API call")
        cudaApi, args = tokens 
        if not type(current_node) in [STCudaLibCall,STCudaKernelCall]:
            new = STCudaLibCall(current_node,current_linemap,current_statement_no)
            new._ignore_in_s2s_translation = not translation_enabled
            new.cudaApi  = cudaApi
            #print("finishes_on_first_line={}".format(finishes_on_first_line))
            assert type(new._parent) in [STModule,STProcedure,STProgram], type(new._parent)
            append_if_not_recording_(new)
    def CudaKernelCall(tokens):
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        nonlocal keep_recording
        log_detection_("CUDA kernel call")
        kernel_name, kernel_launch_args, args = tokens 
        assert type(current_node) in [STModule,STProcedure,STProgram], "type is: "+str(type(current_node))
        new = STCudaKernelCall(current_node,current_linemap,current_statement_no)
        new._ignore_in_s2s_translation = not translation_enabled
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
        new = STAccDirective(current_node,current_linemap,current_statement_no,directive_no)
        new._ignore_in_s2s_translation = not translation_enabled
        directive_no += 1
        # if end directive ascend
        if new.is_end_directive() and\
           type(current_node) is STAccDirective and current_node.is_kernels_directive():
            ascend_()
            current_node.append(new)
        # descend in constructs or new node
        elif new.is_parallel_loop_directive() or new.is_kernels_loop_directive() or\
             (not new.is_end_directive() and new.is_parallel_directive()):
            new = STAccLoopKernel(current_node,current_linemap,current_statement_no,directive_no)
            new.kind = "acc-compute-construct"
            new._ignore_in_s2s_translation = not translation_enabled
            new._do_loop_ctr_memorised=do_loop_ctr
            descend_(new)  # descend also appends 
            keep_recording = True
        elif not new.is_end_directive() and new.is_kernels_directive():
            descend_(new)  # descend also appends 
            #print("descending into kernels or parallel construct")
        else:
           # append new directive
           append_if_not_recording_(new)
    def CufLoopKernel():
        nonlocal translation_enabled
        nonlocal current_node
        nonlocal current_linemap
        nonlocal current_statement_no
        nonlocal do_loop_ctr
        nonlocal keep_recording
        nonlocal directive_no
        log_detection_("CUDA Fortran loop kernel directive")
        new = STCufLoopKernel(current_node,current_linemap,current_statement_no,directive_no)
        new.kind = "cuf-kernel-do"
        new._ignore_in_s2s_translation = not translation_enabled
        new._do_loop_ctr_memorised=do_loop_ctr
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
            parse_result = translator.assignment_begin.parseString(current_statement)
            lvalue = translator.find_first(parse_result,translator.TTLValue)
            if not lvalue is None and lvalue.has_matrix_range_args():
                new  = STAccLoopKernel(current_node,current_linemap,current_statement_no)
                new._ignore_in_s2s_translation = not translation_enabled
                append_if_not_recording_(new)
    def GpufortControl():
        nonlocal current_statement
        nonlocal translation_enabled
        log_detection_("gpufortran control statement")
        if "on" in current_statement:
            translation_enabled = True
        elif "off" in current_statement:
            translation_enabled = False
    
    # TODO completely remove / comment out !$acc end kernels
    module_start.setParseAction(Module_visit)
    program_start.setParseAction(Program_visit)
    function_start.setParseAction(Function_visit)
    subroutine_start.setParseAction(Subroutine_visit)
    
    use.setParseAction(UseStatement)
    CONTAINS.setParseAction(PlaceHolder)
    IMPLICIT.setParseAction(PlaceHolder)
    
    datatype_reg = Regex(r"\s*\b(type\s*\(\s*\w+\s*\)|character|integer|logical|real|complex|double\s+precision)\b") 
    datatype_reg.setParseAction(Declaration)
    
    attributes.setParseAction(Attributes)
    ALLOCATED.setParseAction(Allocated)
    ALLOCATE.setParseAction(Allocate)
    DEALLOCATE.setParseAction(Deallocate)
    memcpy.setParseAction(Memcpy)
    #pointer_assignment.setParseAction(pointer_assignment)
    non_zero_check.setParseAction(non_zero_check)

    # CUDA Fortran 
    cuda_lib_call.setParseAction(CudaLibCall)
    cuf_kernel_call.setParseAction(CudaKernelCall)

    # OpenACC
    ACC_START.setParseAction(AccDirective)
    assignment_begin.setParseAction(Assignment)

    # GPUFORT control
    gpufort_control.setParseAction(GpufortControl)

    current_file = str(fortran_filepath)
    current_node._children.clear()

    def scan_string(expression_name,expression):
        """
        These expressions might be hidden behind a single-line if.
        """
        nonlocal current_linemap
        nonlocal current_statement_no
        nonlocal current_statement

        matched = len(expression.searchString(current_statement,1))
        if matched:
           utils.logging.log_debug3(LOG_PREFIX,"parse_file.scanString","found expression '{}' in line {}: '{}'".format(expression_name,current_linemap["lineno"],current_linemap["lines"][0].rstrip()))
        else:
           utils.logging.log_debug4(LOG_PREFIX,"parse_file.scanString","did not find expression '{}' in line {}: '{}'".format(expression_name,current_linemap["lineno"],current_linemap["lines"][0].rstrip()))
        return matched
    
    def try_to_parse_string(expression_name,expression,parseAll=False):
        """
        These expressions might never be hidden behind a single-line if or might
        never be an argument of another calls.
        No need to check if comments are in the line as the first
        words of the line must match the expression.
        """
        nonlocal current_linemap
        nonlocal current_statement_no
        nonlocal current_statement_strippedNoComments
        
        try:
           expression.parseString(current_statement_strippedNoComments,parseAll)
           utils.logging.log_debug3(LOG_PREFIX,"parse_file.try_to_parse_string","found expression '{}' in line {}: '{}'".format(expression_name,current_linemap["lineno"],current_linemap["lines"][0].rstrip()))
           return True
        except ParseBaseException as e: 
           utils.logging.log_debug4(LOG_PREFIX,"parse_file.try_to_parse_string","did not find expression '{}' in line '{}'".format(expression_name,current_linemap["lines"][0]))
           utils.logging.log_debug5(LOG_PREFIX,"parse_file.try_to_parse_string",str(e))
           return False

    def is_end_statement_(tokens,kind):
        result = tokens[0] == "end"+kind
        if not result and len(tokens):
            result = tokens[0] == "end" and tokens[1] == kind
        return result
    
    p_do_loop_begin = re.compile(r"(\w+:)?\s*do")

    # parser loop
    for current_linemap in linemaps:
        condition1 = current_linemap["is_active"]
        condition2 = len(current_linemap["included_linemaps"]) or not current_linemap["is_preprocessor_directive"]
        if condition1 and condition2:
            for current_statement_no,current_statement in enumerate(current_linemap["statements"]):
                utils.logging.log_debug4(LOG_PREFIX,"parse_file","parsing statement '{}' associated with lines [{},{}]".format(current_statement.rstrip(),\
                    current_linemap["lineno"],current_linemap["lineno"]+len(current_linemap["lines"])-1))
                
                current_tokens                      = utils.parsingutils.tokenize(current_statement.lower())
                current_statement_stripped           = " ".join(current_tokens)
                current_statement_strippedNoComments = current_statement_stripped.split("!")[0]
                if len(current_tokens):
                    # constructs
                    if current_tokens[0] == "end":
                        for kind in ["program","module","subroutine","function"]: # ignore types/interfaces here
                            if is_end_statement_(current_tokens,kind):
                                 End()
                        if is_end_statement_(current_tokens,"do"):
                            DoLoop_leave()
                    if p_do_loop_begin.match(current_statement_strippedNoComments):
                        DoLoop_visit()    
                    # single-statements
                    if not keep_recording:
                        if "acc" in SOURCE_DIALECTS:
                            for comment_char in "!*c":
                                if current_tokens[0:2]==[comment_char+"$","acc"]:
                                    AccDirective()
                                elif current_tokens[0:2]==[comment_char+"$","gpufort"]:
                                    GpufortControl()
                        if "cuf" in SOURCE_DIALECTS:
                            if "attributes" in current_tokens:
                                try_to_parse_string("attributes",attributes)
                            if "cu" in current_statement_strippedNoComments:
                                scan_string("cuda_lib_call",cuda_lib_call)
                            if "<<<" in current_tokens:
                                try_to_parse_string("cuf_kernel_call",cuf_kernel_call)
                            for comment_char in "!*c":
                                if current_tokens[0:2]==[comment_char+"$","cuf"]:
                                    CufLoopKernel()
                        if "=" in current_tokens:
                            if not try_to_parse_string("memcpy",memcpy,parseAll=True):
                                try_to_parse_string("assignment",assignment_begin)
                                scan_string("non_zero_check",non_zero_check)
                        if "allocated" in current_tokens:
                            scan_string("allocated",ALLOCATED)
                        if "deallocate" in current_tokens:
                            try_to_parse_string("deallocate",DEALLOCATE) 
                        if "allocate" in current_tokens:
                            try_to_parse_string("allocate",ALLOCATE) 
                        if "function" in current_tokens:
                            try_to_parse_string("function",function_start)
                        if "subroutine" in current_tokens:
                            try_to_parse_string("subroutine",subroutine_start)
                        # 
                        if current_tokens[0] == "use":
                            try_to_parse_string("use",use)
                        elif current_tokens[0] == "implicit":
                            try_to_parse_string("implicit",IMPLICIT)
                        elif current_tokens[0] == "module":
                            try_to_parse_string("module",module_start)
                        elif current_tokens[0] == "program":
                            try_to_parse_string("program",program_start)
                        elif current_tokens[0] == "return":
                             Return()
                        elif current_tokens[0] in ["character","integer","logical","real","complex","double"]:
                            try_to_parse_string("declaration",datatype_reg)
                            break
                        elif current_tokens[0] == "type" and current_tokens[1] == "(":
                            try_to_parse_string("declaration",datatype_reg)
                    else:
                        current_node.add_linemap(current_linemap)
                        current_node._last_statement_index = current_statement_no

    assert type(current_node) is STRoot
    utils.logging.log_leave_function(LOG_PREFIX,"parse_file")
    return current_node

def postprocess(stree,index,hip_module_suffix):
    """
    Add use statements as well as handles plus their creation and destruction for certain
    math libraries.
    """
    utils.logging.log_enter_function(LOG_PREFIX,"postprocess")
    if "hip" in DESTINATION_DIALECT or len(KERNELS_TO_CONVERT_TO_HIP):
        # todo:
        # for mod/prog in stree:
        #   mod_name <- mod/prog.name
        #   for kernels in body:
        #
        
        # insert use statements at appropriate point
        def is_accelerated(child):
            return isinstance(child,STLoopKernel) or\
                   (type(child) is STProcedure and child.is_kernel_subroutine())
        
        for stmodule in stree.find_all(filter=lambda child: type(child) in [STModule,STProgram],recursively=False):
            module_name = stmodule.name 
            kernels    = stmodule.find_all(filter=is_accelerated, recursively=True)
            for kernel in kernels:
                if "hip" in DESTINATION_DIALECT or\
                  kernel.min_lineno() in kernels_to_convert_to_hip or\
                  kernel.kernel_name() in kernels_to_convert_to_hip:
                    stnode = kernel._parent.find_first(filter=lambda child: type(child) in [STUseStatement,STDeclaration,STPlaceHolder])
                    assert not stnode is None
                    indent = stnode.first_line_indent()
                    stnode.add_to_prolog("{}use {}{}\n".format(indent,module_name,hip_module_suffix))
   
    if "cuf" in SOURCE_DIALECTS:
         _intrnl_postprocess_cuf(stree)
    if "acc" in SOURCE_DIALECTS:
         _intrnl_postprocess_acc(stree)
    utils.logging.log_leave_function(LOG_PREFIX,"postprocess")