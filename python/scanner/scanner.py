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
scannerDir = os.path.dirname(__file__)
exec(open("{0}/scanner_options.py.in".format(scannerDir)).read())
exec(open("{0}/scanner_tree.py.in".format(scannerDir)).read())
exec(open("{0}/openacc/scanner_tree_acc.py.in".format(scannerDir)).read())
exec(open("{0}/cudafortran/scanner_tree_cuf.py.in".format(scannerDir)).read())

def checkDestinationDialect(destinationDialect):
    if destinationDialect in SUPPORTED_DESTINATION_DIALECTS:
        return destinationDialect
    else:
        msg = "scanner: destination dialect '{}' is not supported. Must be one of: {}".format(\
                destinationDialect,", ".join(SUPPORTED_DESTINATION_DIALECTS))
        utils.logging.logError(LOG_PREFIX,"checkDestinationDialect",msg)
        sys.exit(SCANNER_ERROR_CODE)

def _intrnl_postprocessAcc(stree):
    """
    Add use statements as well as handles plus their creation and destruction for certain
    math libraries.
    """
    global LOG_PREFIX
    global DESTINATION_DIALECT
    global RUNTIME_MODULE_NAMES
    
    utils.logging.logEnterFunction(LOG_PREFIX,"__postprocessAcc")
    
    directives = stree.find_all(filter=lambda node: isinstance(node,STAccDirective), recursively=True)
    for directive in directives:
         stnode = directive._parent.find_first(filter=lambda child : not child._ignoreInS2STranslation and type(child) in [STUseStatement,STDeclaration,STPlaceHolder])
         # add acc use statements
         if not stnode is None:
             indent = stnode.firstLineIndent()
             accRuntimeModuleName = RUNTIME_MODULE_NAMES[DESTINATION_DIALECT]
             if accRuntimeModuleName != None and len(accRuntimeModuleName):
                 stnode.addToProlog("{0}use {1}\n{0}use iso_c_binding\n".format(indent,accRuntimeModuleName))
        #if type(directive._parent
    utils.logging.logLeaveFunction(LOG_PREFIX,"__postprocessAcc")
    
def _intrnl_postprocessCuf(stree):
    """
    Add use statements as well as handles plus their creation and destruction for certain
    math libraries.
    """
    global LOG_PREFIX
    global CUBLAS_VERSION 
    utils.logging.logEnterFunction(LOG_PREFIX,"__postprocessCuf")
    # cublas_v1 detection
    if CUBLAS_VERSION == 1:
        def has_cublas_call_(child):
            return type(child) is STCudaLibCall and child.hasCublas()
        cuf_cublas_calls = stree.find_all(filter=has_cublas_call_, recursively=True)
        #print(cuf_cublas_calls)
        for call in cuf_cublas_calls:
            begin = call._parent.findLast(filter=lambda child : type(child) in [STUseStatement,STDeclaration])
            indent = self.firstLineIndent()
            begin.addToEpilog("{0}type(c_ptr) :: hipblasHandle = c_null_ptr\n".format(indent))
            #print(begin._records)       
 
            localCublasCalls = call._parent.find_all(filter=hasCublasCall, recursively=False)
            first = localCublasCalls[0]
            indent = self.firstLineIndent()
            first.addToProlog("{0}hipblasCreate(hipblasHandle)\n".format(indent))
            last = localCublasCalls[-1]
            indent = self.firstLineIndent()
            last.addToEpilog("{0}hipblasDestroy(hipblasHandle)\n".format(indent))
    utils.logging.logLeaveFunction(LOG_PREFIX,"__postprocessCuf")


# API

# Pyparsing actions that create scanner tree (ST)
def parseFile(records,index,fortranFilepath):
    """
    Generate an object tree (OT). 
    """
    utils.logging.logEnterFunction(LOG_PREFIX,"parseFile",
        {"fortranFilepath":fortranFilepath})

    translationEnabled = TRANSLATION_ENABLED_BY_DEFAULT
    
    currentNode   = STRoot()
    doLoopCtr     = 0     
    keepRecording = False
    currentFile   = str(fortranFilepath)
    currentRecord = None
    directiveNo   = 0

    def logDetection_(kind):
        nonlocal currentNode
        nonlocal currentRecord
        utils.logging.logDebug2(LOG_PREFIX,"parseFile","[current-node={}:{}] found {} in line {}: '{}'".format(\
                currentNode.kind,currentNode.name,kind,currentRecord["lineno"],currentRecord["lines"][0].rstrip("\n")))

    def appendIfNotRecording_(new):
        nonlocal currentNode
        nonlocal keepRecording 
        if not keepRecording:
            currentNode.append(new)
    def descend_(new):
        nonlocal currentNode
        nonlocal currentRecord
        nonlocal currentStatementNo
        currentNode.append(new)
        currentNode=new
        
        currentNodeId = currentNode.kind
        if currentNode.name != None:
            currentNodeId += " '"+currentNode.name+"'"
        parentNodeId = currentNode._parent.kind
        if currentNode._parent.name != None:
            parentNodeId += ":"+currentNode._parent.name

        utils.logging.logDebug(LOG_PREFIX,"parseFile","[current-node={0}] enter {1} in line {2}: '{3}'".format(\
          parentNodeId,currentNodeId,currentRecord["lineno"],currentRecord["lines"][0].rstrip("\n")))
    def ascend_():
        nonlocal currentNode
        nonlocal currentFile
        nonlocal currentRecord
        nonlocal currentStatementNo
        assert not currentNode._parent is None, "In file {}: parent of {} is none".format(currentFile,type(currentNode))
        
        currentNodeId = currentNode.kind
        if currentNode.name != None:
            currentNodeId += " '"+currentNode.name+"'"
        parentNodeId = currentNode._parent.kind
        if currentNode._parent.name != None:
            parentNodeId += ":"+currentNode._parent.name
        
        utils.logging.logDebug(LOG_PREFIX,"parseFile","[current-node={0}] leave {1} in line {2}: '{3}'".format(\
          parentNodeId,currentNodeId,currentRecord["lineno"],currentRecord["lines"][0].rstrip("\n")))
        currentNode = currentNode._parent
   
    # parse actions
    def Module_visit(tokens):
        nonlocal translationEnabled
        nonlocal currentNode
        nonlocal currentRecord
        nonlocal currentStatementNo
        logDetection_("module")
        new = STModule(tokens[0],currentNode,currentRecord,currentStatementNo)
        new._ignoreInS2STranslation = not translationEnabled
        descend_(new)
    def Program_visit(tokens):
        nonlocal translationEnabled
        nonlocal currentNode
        nonlocal currentRecord
        nonlocal currentStatementNo
        logDetection_("program")
        new = STProgram(tokens[0],currentNode,currentRecord,currentStatementNo)
        new._ignoreInS2STranslation = not translationEnabled
        descend_(new)
    def Function_visit(tokens):
        nonlocal translationEnabled
        nonlocal currentNode
        nonlocal currentRecord
        nonlocal keepRecording
        nonlocal index
        logDetection_("function")
        new = STProcedure(tokens[1],"function",\
            currentNode,currentRecord,currentStatementNo,index)
        new._ignoreInS2STranslation = not translationEnabled
        keepRecording = new.keepRecording()
        descend_(new)
    def Subroutine_visit(tokens):
        nonlocal translationEnabled
        nonlocal currentNode
        nonlocal currentRecord
        nonlocal keepRecording
        nonlocal index
        logDetection_("subroutine")
        new = STProcedure(tokens[1],"subroutine",\
            currentNode,currentRecord,currentStatementNo,index)
        new._ignoreInS2STranslation = not translationEnabled
        keepRecording = new.keepRecording()
        descend_(new)
    def End():
        nonlocal currentNode
        nonlocal currentRecord
        nonlocal keepRecording
        logDetection_("end of module/program/function/subroutine")
        assert type(currentNode) in [STModule,STProgram,STProcedure], "In file {}: line {}: type is {}".format(currentFile, currentStatementNo, type(currentNode))
        if type(currentNode) is STProcedure and currentNode.mustBeAvailableOnDevice():
            currentNode.addRecord(currentRecord)
            currentNode._lastStatementIndex = currentStatementNo
            currentNode.completeInit()
            keepRecording = False
        if not keepRecording and type(currentNode) in [STProcedure,STProgram]:
            new = STEndOrReturn(currentNode,currentRecord,currentStatementNo)
            appendIfNotRecording_(new)
        ascend_()
    def Return():
        nonlocal currentNode
        nonlocal currentRecord
        nonlocal keepRecording
        logDetection_("return statement")
        if not keepRecording and type(currentNode) in [STProcedure,STProgram]:
            new = STEndOrReturn(currentNode,currentRecord,currentStatementNo)
            appendIfNotRecording_(new)
        ascend_()
    def inKernelsAccRegionAndNotRecording():
        nonlocal currentNode
        nonlocal keepRecording
        return not keepRecording and\
            (type(currentNode) is STAccDirective) and\
            (currentNode.isKernelsDirective())
    def DoLoop_visit():
        nonlocal translationEnabled
        nonlocal currentNode
        nonlocal currentRecord
        nonlocal doLoopCtr
        nonlocal keepRecording
        logDetection_("do loop")
        if inKernelsAccRegionAndNotRecording():
            new = STAccLoopKernel(currentNode,currentRecord,currentStatementNo)
            new._ignoreInS2STranslation = not translationEnabled
            new._doLoopCtrMemorised=doLoopCtr
            descend_(new) 
            keepRecording = True
        doLoopCtr += 1
    def DoLoop_leave():
        nonlocal currentNode
        nonlocal currentRecord
        nonlocal currentStatementNo
        nonlocal doLoopCtr
        nonlocal keepRecording
        logDetection_("end of do loop")
        doLoopCtr -= 1
        if isinstance(currentNode, STLoopKernel):
            if keepRecording and currentNode._doLoopCtrMemorised == doLoopCtr:
                currentNode.addRecord(currentRecord)
                currentNode._lastStatementIndex = currentStatementNo
                currentNode.completeInit()
                ascend_()
                keepRecording = False
    def Declaration():
        nonlocal translationEnabled
        nonlocal currentNode
        nonlocal currentRecord
        nonlocal currentStatementNo
        logDetection_("declaration")
        new = STDeclaration(currentNode,currentRecord,currentStatementNo)
        new._ignoreInS2STranslation = not translationEnabled
        appendIfNotRecording_(new)
    def Attributes(tokens):
        nonlocal translationEnabled
        nonlocal currentNode
        nonlocal currentRecord
        nonlocal currentStatementNo
        logDetection_("attributes statement")
        new = STAttributes(currentNode,currentRecord,currentStatementNo)
        new._ignoreInS2STranslation = not translationEnabled
        currentNode.append(new)
    def UseStatement(tokens):
        nonlocal currentNode
        nonlocal currentRecord
        nonlocal currentStatementNo
        logDetection_("use statement")
        new = STUseStatement(currentNode,currentRecord,currentStatementNo)
        new._ignoreInS2STranslation = not translationEnabled
        new.name = translator.make_f_str(tokens[1]) # just get the name, ignore specific includes
        appendIfNotRecording_(new)
    def PlaceHolder(tokens):
        nonlocal translationEnabled
        nonlocal currentNode
        nonlocal currentRecord
        nonlocal currentStatementNo
        logDetection_("placeholder")
        new = STPlaceHolder(currentNode,currentRecord,currentStatementNo)
        new._ignoreInS2STranslation = not translationEnabled
        appendIfNotRecording_(new)
    def NonZeroCheck(tokens):
        nonlocal translationEnabled
        nonlocal currentNode
        nonlocal currentRecord
        nonlocal currentStatementNo
        logDetection_("non-zero check")
        new = STNonZeroCheck(currentNode,currentRecord,currentStatementNo)
        new._ignoreInS2STranslation = not translationEnabled
        appendIfNotRecording_(new)
    def Allocated(tokens):
        nonlocal currentNode
        nonlocal translationEnabled
        nonlocal currentRecord
        nonlocal currentStatementNo
        logDetection_("allocated statement")
        new = STAllocated(currentNode,currentRecord,currentStatementNo)
        new._ignoreInS2STranslation = not translationEnabled
        appendIfNotRecording_(new)
    def Allocate(tokens):
        nonlocal translationEnabled
        nonlocal currentNode
        nonlocal currentRecord
        nonlocal currentStatementNo
        logDetection_("allocate statement")
        new = STAllocate(currentNode,currentRecord,currentStatementNo)
        new._ignoreInS2STranslation = not translationEnabled
        appendIfNotRecording_(new)
    def Deallocate(tokens):
        nonlocal translationEnabled
        nonlocal currentNode
        nonlocal currentRecord
        nonlocal currentStatementNo
        logDetection_("deallocate statement")
        # TODO filter variable, replace with hipFree
        new = STDeallocate(currentNode,currentRecord,currentStatementNo)
        new._ignoreInS2STranslation = not translationEnabled
        appendIfNotRecording_(new)
    def Memcpy(tokens):
        nonlocal translationEnabled
        nonlocal currentNode
        nonlocal currentRecord
        nonlocal currentStatementNo
        logDetection_("memcpy")
        new = STMemcpy(currentNode,currentRecord,currentStatementNo)
        new._ignoreInS2STranslation = not translationEnabled
        appendIfNotRecording_(new)
    def CudaLibCall(tokens):
        #TODO scan for cudaMemcpy calls
        nonlocal translationEnabled
        nonlocal currentNode
        nonlocal currentRecord
        nonlocal currentStatementNo
        nonlocal keepRecording
        logDetection_("CUDA API call")
        cudaApi, args = tokens 
        if not type(currentNode) in [STCudaLibCall,STCudaKernelCall]:
            new = STCudaLibCall(currentNode,currentRecord,currentStatementNo)
            new._ignoreInS2STranslation = not translationEnabled
            new.cudaApi  = cudaApi
            #print("finishesOnFirstLine={}".format(finishesOnFirstLine))
            assert type(new._parent) in [STModule,STProcedure,STProgram], type(new._parent)
            appendIfNotRecording_(new)
    def CudaKernelCall(tokens):
        nonlocal translationEnabled
        nonlocal currentNode
        nonlocal currentRecord
        nonlocal currentStatementNo
        nonlocal keepRecording
        logDetection_("CUDA kernel call")
        kernelName, kernel_launch_args, args = tokens 
        assert type(currentNode) in [STModule,STProcedure,STProgram], "type is: "+str(type(currentNode))
        new = STCudaKernelCall(currentNode,currentRecord,currentStatementNo)
        new._ignoreInS2STranslation = not translationEnabled
        appendIfNotRecording_(new)
    def AccDirective():
        nonlocal translationEnabled
        nonlocal currentNode
        nonlocal currentRecord
        nonlocal currentStatementNo
        nonlocal keepRecording
        nonlocal doLoopCtr
        nonlocal directiveNo
        logDetection_("OpenACC directive")
        new = STAccDirective(currentNode,currentRecord,currentStatementNo,directiveNo)
        new._ignoreInS2STranslation = not translationEnabled
        directiveNo += 1
        # if end directive ascend
        if new.isEndDirective() and\
           type(currentNode) is STAccDirective and currentNode.isKernelsDirective():
            ascend_()
            currentNode.append(new)
        # descend in constructs or new node
        elif new.isParallelLoopDirective() or new.isKernelsLoopDirective() or\
             (not new.isEndDirective() and new.isParallelDirective()):
            new = STAccLoopKernel(currentNode,currentRecord,currentStatementNo,directiveNo)
            new.kind = "acc-compute-construct"
            new._ignoreInS2STranslation = not translationEnabled
            new._doLoopCtrMemorised=doLoopCtr
            descend_(new)  # descend also appends 
            keepRecording = True
        elif not new.isEndDirective() and new.isKernelsDirective():
            descend_(new)  # descend also appends 
            #print("descending into kernels or parallel construct")
        else:
           # append new directive
           appendIfNotRecording_(new)
    def CufLoopKernel():
        nonlocal translationEnabled
        nonlocal currentNode
        nonlocal currentRecord
        nonlocal currentStatementNo
        nonlocal doLoopCtr
        nonlocal keepRecording
        nonlocal directiveNo
        logDetection_("CUDA Fortran loop kernel directive")
        new = STCufLoopKernel(currentNode,currentRecord,currentStatementNo,directiveNo)
        new.kind = "cuf-kernel-do"
        new._ignoreInS2STranslation = not translationEnabled
        new._doLoopCtrMemorised=doLoopCtr
        directiveNo += 1
        descend_(new) 
        keepRecording = True
    def Assignment(tokens):
        nonlocal currentNode
        nonlocal translationEnabled
        nonlocal currentRecord
        nonlocal currentStatementNo
        nonlocal currentStatement
        logDetection_("assignment")
        if inKernelsAccRegionAndNotRecording():
            parseResult = translator.assignment_begin.parseString(currentStatement)
            lvalue = translator.find_first(parseResult,translator.TTLValue)
            if not lvalue is None and lvalue.hasMatrixRangeArgs():
                new  = STAccLoopKernel(currentNode,currentRecord,currentStatementNo)
                new._ignoreInS2STranslation = not translationEnabled
                appendIfNotRecording_(new)
    def GpufortControl():
        nonlocal currentStatement
        nonlocal translationEnabled
        logDetection_("gpufortran control statement")
        if "on" in currentStatement:
            translationEnabled = True
        elif "off" in currentStatement:
            translationEnabled = False
    
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

    currentFile = str(fortranFilepath)
    currentNode._children.clear()

    def scanString(expressionName,expression):
        """
        These expressions might be hidden behind a single-line if.
        """
        nonlocal currentRecord
        nonlocal currentStatementNo
        nonlocal currentStatement

        matched = len(expression.searchString(currentStatement,1))
        if matched:
           utils.logging.logDebug3(LOG_PREFIX,"parseFile.scanString","found expression '{}' in line {}: '{}'".format(expressionName,currentRecord["lineno"],currentRecord["lines"][0].rstrip()))
        else:
           utils.logging.logDebug4(LOG_PREFIX,"parseFile.scanString","did not find expression '{}' in line {}: '{}'".format(expressionName,currentRecord["lineno"],currentRecord["lines"][0].rstrip()))
        return matched
    
    def tryToParseString(expressionName,expression,parseAll=False):
        """
        These expressions might never be hidden behind a single-line if or might
        never be an argument of another calls.
        No need to check if comments are in the line as the first
        words of the line must match the expression.
        """
        nonlocal currentRecord
        nonlocal currentStatementNo
        nonlocal currentStatementStrippedNoComments
        
        try:
           expression.parseString(currentStatementStrippedNoComments,parseAll)
           utils.logging.logDebug3(LOG_PREFIX,"parseFile.tryToParseString","found expression '{}' in line {}: '{}'".format(expressionName,currentRecord["lineno"],currentRecord["lines"][0].rstrip()))
           return True
        except ParseBaseException as e: 
           utils.logging.logDebug4(LOG_PREFIX,"parseFile.tryToParseString","did not find expression '{}' in line '{}'".format(expressionName,currentRecord["lines"][0]))
           utils.logging.logDebug5(LOG_PREFIX,"parseFile.tryToParseString",str(e))
           return False

    def isEndStatement_(tokens,kind):
        result = tokens[0] == "end"+kind
        if not result and len(tokens):
            result = tokens[0] == "end" and tokens[1] == kind
        return result
    
    pDoLoopBegin = re.compile(r"(\w+:)?\s*do")

    # parser loop
    for currentRecord in records:
        condition1 = currentRecord["is_active"]
        condition2 = len(currentRecord["included_records"]) or not currentRecord["is_preprocessor_directive"]
        if condition1 and condition2:
            for currentStatementNo,currentStatement in enumerate(currentRecord["statements"]):
                utils.logging.logDebug4(LOG_PREFIX,"parseFile","parsing statement '{}' associated with lines [{},{}]".format(currentStatement.rstrip(),\
                    currentRecord["lineno"],currentRecord["lineno"]+len(currentRecord["lines"])-1))
                
                currentTokens                      = utils.pyparsingutils.tokenize(currentStatement.lower())
                currentStatementStripped           = " ".join(currentTokens)
                currentStatementStrippedNoComments = currentStatementStripped.split("!")[0]
                if len(currentTokens):
                    # constructs
                    if currentTokens[0] == "end":
                        for kind in ["program","module","subroutine","function"]: # ignore types/interfaces here
                            if isEndStatement_(currentTokens,kind):
                                 End()
                        if isEndStatement_(currentTokens,"do"):
                            DoLoop_leave()
                    if pDoLoopBegin.match(currentStatementStrippedNoComments):
                        DoLoop_visit()    
                    # single-statements
                    if not keepRecording:
                        if "acc" in SOURCE_DIALECTS:
                            for commentChar in "!*c":
                                if currentTokens[0:2]==[commentChar+"$","acc"]:
                                    AccDirective()
                                elif currentTokens[0:2]==[commentChar+"$","gpufort"]:
                                    GpufortControl()
                        if "cuf" in SOURCE_DIALECTS:
                            if "attributes" in currentTokens:
                                tryToParseString("attributes",attributes)
                            if "cu" in currentStatementStrippedNoComments:
                                scanString("cuda_lib_call",cuda_lib_call)
                            if "<<<" in currentTokens:
                                tryToParseString("cuf_kernel_call",cuf_kernel_call)
                            for commentChar in "!*c":
                                if currentTokens[0:2]==[commentChar+"$","cuf"]:
                                    CufLoopKernel()
                        if "=" in currentTokens:
                            if not tryToParseString("memcpy",memcpy,parseAll=True):
                                tryToParseString("assignment",assignment_begin)
                                scanString("non_zero_check",non_zero_check)
                        if "allocated" in currentTokens:
                            scanString("allocated",ALLOCATED)
                        if "deallocate" in currentTokens:
                            tryToParseString("deallocate",DEALLOCATE) 
                        if "allocate" in currentTokens:
                            tryToParseString("allocate",ALLOCATE) 
                        if "function" in currentTokens:
                            tryToParseString("function",function_start)
                        if "subroutine" in currentTokens:
                            tryToParseString("subroutine",subroutine_start)
                        # 
                        if currentTokens[0] == "use":
                            tryToParseString("use",use)
                        elif currentTokens[0] == "implicit":
                            tryToParseString("implicit",IMPLICIT)
                        elif currentTokens[0] == "module":
                            tryToParseString("module",module_start)
                        elif currentTokens[0] == "program":
                            tryToParseString("program",program_start)
                        elif currentTokens[0] == "return":
                             Return()
                        elif currentTokens[0] in ["character","integer","logical","real","complex","double"]:
                            tryToParseString("declaration",datatype_reg)
                            break
                        elif currentTokens[0] == "type" and currentTokens[1] == "(":
                            tryToParseString("declaration",datatype_reg)
                    else:
                        currentNode.addRecord(currentRecord)
                        currentNode._lastStatementIndex = currentStatementNo

    assert type(currentNode) is STRoot
    utils.logging.logLeaveFunction(LOG_PREFIX,"parseFile")
    return currentNode

def postprocess(stree,index,hipModuleSuffix):
    """
    Add use statements as well as handles plus their creation and destruction for certain
    math libraries.
    """
    utils.logging.logEnterFunction(LOG_PREFIX,"postprocess")
    if "hip" in DESTINATION_DIALECT or len(KERNELS_TO_CONVERT_TO_HIP):
        # todo:
        # for mod/prog in stree:
        #   modName <- mod/prog.name
        #   for kernels in body:
        #
        
        # insert use statements at appropriate point
        def isAccelerated(child):
            return isinstance(child,STLoopKernel) or\
                   (type(child) is STProcedure and child.isKernelSubroutine())
        
        for stmodule in stree.find_all(filter=lambda child: type(child) in [STModule,STProgram],recursively=False):
            moduleName = stmodule.name 
            kernels    = stmodule.find_all(filter=isAccelerated, recursively=True)
            for kernel in kernels:
                if "hip" in DESTINATION_DIALECT or\
                  kernel.minLineno() in kernelsToConvertToHip or\
                  kernel.kernelName() in kernelsToConvertToHip:
                    stnode = kernel._parent.find_first(filter=lambda child: type(child) in [STUseStatement,STDeclaration,STPlaceHolder])
                    assert not stnode is None
                    indent = stnode.firstLineIndent()
                    stnode.addToProlog("{}use {}{}\n".format(indent,moduleName,hipModuleSuffix))
   
    if "cuf" in SOURCE_DIALECTS:
         _intrnl_postprocessCuf(stree)
    if "acc" in SOURCE_DIALECTS:
         _intrnl_postprocessAcc(stree)
    utils.logging.logLeaveFunction(LOG_PREFIX,"postprocess")
