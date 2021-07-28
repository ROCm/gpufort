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

def __postprocessAcc(stree,hipModuleName):
    """
    Add use statements as well as handles plus their creation and destruction for certain
    math libraries.
    """
    global LOG_PREFIX
    global DESTINATION_DIALECT
    global RUNTIME_MODULE_NAMES
    
    utils.logging.logEnterFunction(LOG_PREFIX,"__postprocessAcc",{"hipModuleName":hipModuleName})
    
    directives = stree.findAll(filter=lambda node: isinstance(node,STAccDirective), recursively=True)
    for directive in directives:
         stnode = directive._parent.findFirst(filter=lambda child : not child._ignoreInS2STranslation and type(child) in [STUseStatement,STDeclaration,STPlaceHolder])
         # add acc use statements
         if not stnode is None:
             indent = stnode.firstLineIndent()
             accRuntimeModuleName = RUNTIME_MODULE_NAMES[DESTINATION_DIALECT]
             if accRuntimeModuleName != None and len(accRuntimeModuleName):
                 stnode.addToProlog("{0}use {1}\n".format(indent,accRuntimeModuleName))
        #if type(directive._parent
    utils.logging.logLeaveFunction(LOG_PREFIX,"__postprocessAcc")
    
def __postprocessCuf(stree,hipModuleName):
    """
    Add use statements as well as handles plus their creation and destruction for certain
    math libraries.
    """
    global LOG_PREFIX
    global CUBLAS_VERSION 
    utils.logging.logEnterFunction(LOG_PREFIX,"__postprocessCuf",{"hipModuleName":hipModuleName})
    # cublas_v1 detection
    if CUBLAS_VERSION == 1:
        def hasCublasCall(child):
            return type(child) is STCudaLibCall and child.hasCublas()
        cublasCalls = stree.findAll(filter=hasCublasCall, recursively=True)
        #print(cublasCalls)
        for call in cublasCalls:
            begin = call._parent.findLast(filter=lambda child : type(child) in [STUseStatement,STDeclaration])
            indent = self.firstLineIndent()
            begin.addToEpilog("{0}type(c_ptr) :: hipblasHandle = c_null_ptr\n".format(indent))
            #print(begin._records)       
 
            localCublasCalls = call._parent.findAll(filter=hasCublasCall, recursively=False)
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
                currentNode._kind,currentNode._name,kind,currentRecord["lineno"],currentRecord["lines"][0]))

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
        
        currentNodeId = currentNode._kind
        if currentNode._name != None:
            currentNodeId += " '"+currentNode._name+"'"
        parentNodeId = currentNode._parent._kind
        if currentNode._parent._name != None:
            parentNodeId += ":"+currentNode._parent._name

        utils.logging.logDebug(LOG_PREFIX,"parseFile","[current-node={0}] enter {1} in line {2}: '{3}'".format(\
          parentNodeId,currentNodeId,currentRecord["lineno"],currentRecord["lines"][0]))
    def ascend_():
        nonlocal currentNode
        nonlocal currentFile
        nonlocal currentRecord
        nonlocal currentStatementNo
        assert not currentNode._parent is None, "In file {}: parent of {} is none".format(currentFile,type(currentNode))
        
        currentNodeId = currentNode._kind
        if currentNode._name != None:
            currentNodeId += " '"+currentNode._name+"'"
        parentNodeId = currentNode._parent._kind
        if currentNode._parent._name != None:
            parentNodeId += ":"+currentNode._parent._name
        
        utils.logging.logDebug(LOG_PREFIX,"parseFile","[current-node={0}] leave {1} in line {2}: '{3}'".format(\
          parentNodeId,currentNodeId,currentRecord["lineno"],currentRecord["lines"][0]))
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
        try:
           new = STProcedure(tokens[1],"subroutine",\
               currentNode,currentRecord,currentStatementNo,index)
           new._ignoreInS2STranslation = not translationEnabled
           keepRecording = new.keepRecording()
        except Exception as e:
            print(e)
        descend_(new)
    def Structure_leave():
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
        new._name = translator.makeFStr(tokens[1]) # just get the name, ignore specific includes
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
        cudaApi, args, finishesOnFirstLine = tokens 
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
        kernelName, kernelLaunchArgs, args, finishesOnFirstLine = tokens 
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
            new._kind = "acc-compute-construct"
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
        new._kind = "cuf-kernel-do"
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
            parseResult = translator.assignmentBegin.parseString(currentStatement)
            lvalue = translator.findFirst(parseResult,translator.TTLValue)
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
    moduleStart.setParseAction(Module_visit)
    programStart.setParseAction(Program_visit)
    functionStart.setParseAction(Function_visit)
    subroutineStart.setParseAction(Subroutine_visit)
    
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
    #pointerAssignment.setParseAction(PointerAssignment)
    nonZeroCheck.setParseAction(NonZeroCheck)

    # CUDA Fortran 
    cudaLibCall.setParseAction(CudaLibCall)
    cudaKernelCall.setParseAction(CudaKernelCall)

    # OpenACC
    ACC_START.setParseAction(AccDirective)
    assignmentBegin.setParseAction(Assignment)

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
    
    def tryToParseString(expressionName,expression):
        """
        These expressions might never be hidden behind a single-line if or might
        never be an argument of another calls.
        No need to check if comments are in the line as the first
        words of the line must match the expression.
        """
        nonlocal currentRecord
        nonlocal currentStatementNo
        nonlocal currentStatement
        
        try:
           expression.parseString(currentStatement)
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

    # parser loop
    for currentRecord in records:
        condition1 = currentRecord["isActive"]
        condition2 = len(currentRecord["includedRecords"]) or not currentRecord["isPreprocessorDirective"]
        if condition1 and condition2:
            for currentStatementNo,currentStatement in enumerate(currentRecord["statements"]):
                utils.logging.logDebug4(LOG_PREFIX,"parseFile","parsing statement '{}' associated with lines [{},{}]".format(currentStatement.rstrip(),\
                    currentRecord["lineno"],currentRecord["lineno"]+len(currentRecord["lines"])-1))
                
                currentTokens = re.split(r"\s+|\t+",currentStatement.lower().strip(" \t"))
                currentStatementStripped           = "".join(currentTokens)
                currentStatementStrippedNoComments = currentStatementStripped.split("!")[0]
                if len(currentTokens):
                    if currentTokens[0].startswith("end"):
                        for kind in ["program","module","subroutine","function","type"]:
                            if isEndStatement_(currentTokens,kind):
                                 Structure_leave()
                        if isEndStatement_(currentTokens,"do"):
                            DoLoop_leave()
                    if "acc" in SOURCE_DIALECTS:
                        for commentChar in "!*c":
                            if currentStatementStripped.startswith(commentChar+"$acc"):
                                AccDirective()
                            if currentStatementStripped.startswith(commentChar+"$gpufort"):
                                GpufortControl()
                    if "cuf" in SOURCE_DIALECTS:
                        if "attributes" in currentStatementStrippedNoComments:
                            tryToParseString("attributes",attributes)
                        if "cu" in currentStatementStrippedNoComments:
                            scanString("cudaLibCall",cudaLibCall)
                        if "<<<" in currentStatementStrippedNoComments:
                            tryToParseString("cudaKernelCall",cudaKernelCall)
                        for commentChar in "!*c":
                            if currentStatementStripped.startswith(commentChar+"$cuf"):
                                CufLoopKernel()
                    if "=" in currentStatementStrippedNoComments:
                        if not tryToParseString("memcpy",memcpy):
                            tryToParseString("assignment",assignmentBegin)
                            tryToParseString("nonZeroCheck",nonZeroCheck)
                    if "allocated" in currentStatementStrippedNoComments:
                        tryToParseString("allocated",ALLOCATED)
                    elif "deallocate" in currentStatementStrippedNoComments:
                        tryToParseString("deallocate",DEALLOCATE) 
                    elif "allocate" in currentStatementStrippedNoComments:
                        tryToParseString("allocate",ALLOCATE) 
                    
                    pDoLoopBegin = re.compile(r"(\w+:)?\s*do")
                    if pDoLoopBegin.match(currentStatementStripped):
                        DoLoop_visit()    
                    
                    if currentTokens[0] == "use":
                        tryToParseString("use",use)
                    elif currentTokens[0] == "implicit":
                        tryToParseString("implicit",IMPLICIT)
                    elif currentTokens[0] == "module":
                        tryToParseString("module",moduleStart)
                    elif currentTokens[0] == "program":
                        tryToParseString("program",programStart)
                    if "function" in currentTokens:
                        tryToParseString("function",functionStart)
                    if "subroutine" in currentTokens:
                        tryToParseString("subroutine",subroutineStart)
                    for expr in ["type","character","integer","logical","real","complex","double"]:
                        if expr in currentTokens[0]:
                            tryToParseString("declaration",datatype_reg)
                            break 
                    if keepRecording:
                       try:
                          currentNode.addRecord(currentRecord)
                          currentNode._lastStatementIndex = currentStatementNo
                       except Exception as e:
                          utils.logging.logError(LOG_PREFIX,"parseFile","While parsing file {}".format(currentFile))
                          raise e
    assert type(currentNode) is STRoot
    utils.logging.logLeaveFunction(LOG_PREFIX,"parseFile")
    return currentNode

def postprocess(stree,hipModuleName,index):
    """
    Add use statements as well as handles plus their creation and destruction for certain
    math libraries.
    """
    utils.logging.logEnterFunction(LOG_PREFIX,"postprocess",{"hipModuleName":hipModuleName})
    if "hip" in DESTINATION_DIALECT or len(KERNELS_TO_CONVERT_TO_HIP):
        # insert use kernel statements at appropriate point
        def isLoopKernel(child):
            return isinstance(child,STLoopKernel) or\
                   (type(child) is STProcedure and child.isKernelSubroutine())
        kernels = stree.findAll(filter=isLoopKernel, recursively=True)
        for kernel in kernels:
            if "hip" in DESTINATION_DIALECT or\
              kernel.minLineno() in kernelsToConvertToHip or\
              kernel.kernelName() in kernelsToConvertToHip:
                stnode = kernel._parent.findFirst(filter=lambda child: type(child) in [STUseStatement,STDeclaration,STPlaceHolder])
                assert not stnode is None
                indent = stnode.firstLineIndent()
                stnode.addToProlog("{0}use {1}\n".format(indent,hipModuleName))
   
    if "cuf" in SOURCE_DIALECTS:
         __postprocessCuf(stree,hipModuleName)
    if "acc" in SOURCE_DIALECTS:
         __postprocessAcc(stree,hipModuleName)
    utils.logging.logLeaveFunction(LOG_PREFIX,"postprocess")
