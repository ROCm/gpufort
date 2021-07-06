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
exec(open("{0}/scanner_groups.py.in".format(scannerDir)).read())

def checkDestinationDialect(destinationDialect):
    if destinationDialect in SUPPORTED_DESTINATION_DIALECTS:
        return destinationDialect
    else:
        msg = "scanner: destination dialect '{}' is not supported. Must be one of: {}".format(\
                destinationDialect,", ".join(SUPPORTED_DESTINATION_DIALECTS))
        utils.logging.logError(LOG_PREFIX,"checkDestinationDialect",msg)
        sys.exit(SCANNER_ERROR_CODE)

def handleIncludeStatements(fortranFilepath,lines):
    """
    Copy included files' content into currentNode file.
    """
    global LOG_PREFIX    
    utils.logging.logEnterFunction(LOG_PREFIX,"handleIncludeStatements",\
      {"fortranFilepath":fortranFilepath})

    def processLine(line):
        if "#include" in line.lower():
            relativeSnippetPath = line.split(' ')[-1].replace('\"','').replace('\'','').strip()
            snippetPath = os.path.dirname(fortranFilepath) + "/" + relativeSnippetPath
            try:
                result = ["! {stmt}\n".format(stmt=line)]
                result.append("! begin gpufort include\n")
                result += open(snippetPath,"r").readlines()
                result.append("! end gpufort include\n")
                print("processed include: "+line.strip())
                return result
            except Exception as e:
                raise e
                return [line]
        else:
            return [line]
     
    result = []
    for line in lines:
        result += processLine(line)
    utils.logging.logLeaveFunction(LOG_PREFIX,"handleIncludeStatements")
    return result

def __postprocessAcc(stree,hipModuleName):
    """
    Add use statements as well as handles plus their creation and destruction for certain
    math libraries.
    """
    global LOG_PREFIX
    global DESTINATION_DIALECT
    global RUNTIME_MODULE_NAMES
    
    utils.logging.logEnterFunction(LOG_PREFIX,"__postprocessAcc",{"hipModuleName":hipModuleName})
    
    # acc detection
    directives = stree.findAll(filter=lambda node: isinstance(node,STAccDirective), recursively=True)
    for directive in directives:
         stnode = directive._parent.findFirst(filter=lambda child : type(child) in [STUseStatement,STDeclaration,STPlaceHolder])
         if not stnode is None:
             indent = " "*(len(stnode.lines()[0]) - len(stnode.lines()[0].lstrip()))
             accRuntimeModuleName = RUNTIME_MODULE_NAMES[DESTINATION_DIALECT]
             if accRuntimeModuleName != None and len(accRuntimeModuleName):
                 stnode._preamble.add("{0}use iso_c_binding\n{0}use {1}\n".format(indent,accRuntimeModuleName))
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
            indent = " "*(len(begin.lines()[0]) - len(begin.lines()[0].lstrip()))
            begin._epilog.add("{0}type(c_ptr) :: hipblasHandle = c_null_ptr\n".format(indent))
            #print(begin._lines)       
 
            localCublasCalls = call._parent.findAll(filter=hasCublasCall, recursively=False)
            first = localCublasCalls[0]
            indent = " "*(len(first.lines()[0]) - len(first.lines()[0].lstrip()))
            first._preamble.add("{0}hipblasCreate(hipblasHandle)\n".format(indent))
            last = localCublasCalls[-1]
            indent = " "*(len(last.lines()[0]) - len(last.lines()[0].lstrip()))
            last._epilog.add("{0}hipblasDestroy(hipblasHandle)\n".format(indent))
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
    currentLineno = -1
    currentLines  = []
    directiveNo   = 0

    def logDetection_(kind):
        nonlocal currentNode
        nonlocal currentLineno
        nonlocal currentLines
        utils.logging.logDebug2(LOG_PREFIX,".__parseFile","[current-node={}:{}] found {} in line {}: '{}'".format(\
                currentNode._kind,currentNode._name,kind,currentLineno+1,currentLines[0]))

    def appendIfNotRecording_(new):
        nonlocal currentNode
        nonlocal keepRecording 
        if not keepRecording:
            currentNode.append(new)
    def descend_(new):
        nonlocal currentNode
        nonlocal currentLineno
        nonlocal currentLines
        currentNode.append(new)
        currentNode=new
        
        currentNodeId = currentNode._kind
        if currentNode._name != None:
            currentNodeId += " '"+currentNode._name+"'"
        parentNodeId = currentNode._parent._kind
        if currentNode._parent._name != None:
            parentNodeId += ":"+currentNode._parent._name

        utils.logging.logDebug(LOG_PREFIX,"parseFile","[current-node={0}] enter {1} in line {2}: '{3}'".format(\
          parentNodeId,currentNodeId,currentLineno+1,currentLines[0]))
    def ascend_():
        nonlocal currentNode
        nonlocal currentFile
        nonlocal currentLineno
        nonlocal currentLines
        assert not currentNode._parent is None, "In file {}: parent of {} is none".format(currentFile,type(currentNode))
        
        currentNodeId = currentNode._kind
        if currentNode._name != None:
            currentNodeId += " '"+currentNode._name+"'"
        parentNodeId = currentNode._parent._kind
        if currentNode._parent._name != None:
            parentNodeId += ":"+currentNode._parent._name
        
        utils.logging.logDebug(LOG_PREFIX,"parseFile","[current-node={0}] leave {1} in line {2}: '{3}'".format(\
          parentNodeId,currentNodeId,currentLineno+1,currentLines[0]))
        currentNode = currentNode._parent
   
    # parse actions
    def Module_visit(tokens):
        nonlocal currentNode
        nonlocal translationEnabled
        nonlocal currentLineno
        nonlocal currentLines
        logDetection_("module")
        new = STModule(tokens[0],currentNode,currentLineno)
        new._ignoreInS2STranslation = not translationEnabled
        descend_(new)
    def Program_visit(tokens):
        nonlocal currentNode
        nonlocal translationEnabled
        nonlocal currentLineno
        nonlocal currentLines
        logDetection_("program")
        new = STProgram(tokens[0],currentNode,currentLineno)
        new._ignoreInS2STranslation = not translationEnabled
        descend_(new)
    def Function_visit(tokens):
        nonlocal currentNode
        nonlocal translationEnabled
        nonlocal keepRecording
        nonlocal index
        logDetection_("function")
        new = STProcedure(tokens[1],"function",index,\
          parent=currentNode,lineno=currentLineno,lines=currentLines)
        new._ignoreInS2STranslation = not translationEnabled
        keepRecording = new.keepRecording()
        if keepRecording:
            new._lines = []
        descend_(new)
    def Subroutine_visit(tokens):
        nonlocal currentNode
        nonlocal translationEnabled
        nonlocal keepRecording
        nonlocal index
        logDetection_("subroutine")
        new = STProcedure(tokens[1],"subroutine",index,\
          parent=currentNode,lineno=currentLineno,lines=currentLines)
        new._ignoreInS2STranslation = not translationEnabled
        keepRecording = new.keepRecording()
        if keepRecording:
            new._lines = []
        descend_(new)
    def Structure_leave(tokens):
        nonlocal currentNode
        nonlocal keepRecording
        logDetection_("end of module/program/function/subroutine")
        assert type(currentNode) in [STModule,STProgram,STProcedure], "In file {}: line {}: type is {}".format(currentFile, currentLineno, type(currentNode))
        if type(currentNode) is STProcedure and currentNode.mustBeAvailableOnDevice():
            currentNode._lines += currentLines
            keepRecording = False
        ascend_()
    def inKernelsAccRegionAndNotRecording():
        nonlocal currentNode
        nonlocal keepRecording
        return not keepRecording and\
            (type(currentNode) is STAccDirective) and\
            (currentNode.isKernelsDirective())
    def DoLoop_visit(tokens):
        nonlocal keepRecording
        nonlocal translationEnabled
        nonlocal doLoopCtr
        logDetection_("do loop")
        if inKernelsAccRegionAndNotRecording():
            new = STAccLoopKernel(parent=currentNode,lineno=currentLineno,lines=currentLines)
            new._ignoreInS2STranslation = not translationEnabled
            new._lines = []
            new._doLoopCtrMemorised=doLoopCtr
            descend_(new) 
            keepRecording = True
        doLoopCtr += 1
    def DoLoop_leave(tokens):
        nonlocal currentNode
        nonlocal currentLines
        nonlocal doLoopCtr
        nonlocal keepRecording
        logDetection_("end of do loop")
        doLoopCtr -= 1
        if isinstance(currentNode, STLoopKernel):
            if keepRecording and currentNode._doLoopCtrMemorised == doLoopCtr:
                currentNode._lines += currentLines
                ascend_()
                keepRecording = False
    def Declaration(tokens):
        nonlocal currentNode
        nonlocal translationEnabled
        nonlocal currentLineno
        nonlocal currentLines
        logDetection_("declaration")
        new = STDeclaration(parent=currentNode,lineno=currentLineno,lines=currentLines)
        new._ignoreInS2STranslation = not translationEnabled
        appendIfNotRecording_(new)
    def Attributes(tokens):
        nonlocal currentNode
        nonlocal translationEnabled
        nonlocal currentLineno
        nonlocal currentLines
        logDetection_("attributes statement")
        new = STAttributes(parent=currentNode,lineno=currentLineno,lines=currentLines)
        new._ignoreInS2STranslation = not translationEnabled
        currentNode.append(new)
    def UseStatement(tokens):
        nonlocal currentNode
        nonlocal currentLineno
        nonlocal currentLines
        logDetection_("use statement")
        new = STUseStatement(parent=currentNode,lineno=currentLineno,lines=currentLines)
        new._ignoreInS2STranslation = not translationEnabled
        new._name = translator.makeFStr(tokens[1]) # just get the name, ignore specific includes
        appendIfNotRecording_(new)
    def PlaceHolder(tokens):
        nonlocal currentNode
        nonlocal translationEnabled
        nonlocal currentLineno
        nonlocal currentLines
        logDetection_("placeholder")
        new = STPlaceHolder(currentNode,currentLineno,currentLines)
        new._ignoreInS2STranslation = not translationEnabled
        appendIfNotRecording_(new)
    def NonZeroCheck(tokens):
        nonlocal currentNode
        nonlocal translationEnabled
        nonlocal currentLineno
        nonlocal currentLines
        logDetection_("non-zero check")
        new=STNonZeroCheck(parent=currentNode,lineno=currentLineno,lines=currentLines)
        new._ignoreInS2STranslation = not translationEnabled
        appendIfNotRecording_(new)
    def Allocated(tokens):
        nonlocal currentNode
        nonlocal translationEnabled
        nonlocal currentLineno
        nonlocal currentLines
        logDetection_("allocated statement")
        new=STAllocated(parent=currentNode,lineno=currentLineno,lines=currentLines)
        new._ignoreInS2STranslation = not translationEnabled
        appendIfNotRecording_(new)
    def Allocate(tokens):
        nonlocal currentNode
        nonlocal translationEnabled
        nonlocal currentLineno
        nonlocal currentLines
        logDetection_("allocate statement")
        new=STAllocate(parent=currentNode,lineno=currentLineno,lines=currentLines)
        new._ignoreInS2STranslation = not translationEnabled
        appendIfNotRecording_(new)
    def Deallocate(tokens):
        nonlocal currentNode
        nonlocal translationEnabled
        nonlocal currentLineno
        nonlocal currentLines
        logDetection_("deallocate statement")
        # TODO filter variable, replace with hipFree
        new = STDeallocate(parent=currentNode,lineno=currentLineno,lines=currentLines)
        new._ignoreInS2STranslation = not translationEnabled
        appendIfNotRecording_(new)
    def Memcpy(tokens):
        nonlocal currentNode
        nonlocal translationEnabled
        nonlocal currentLineno
        nonlocal currentLines
        logDetection_("memcpy")
        new = STMemcpy(parent=currentNode,lineno=currentLineno,lines=currentLines)
        new._ignoreInS2STranslation = not translationEnabled
        appendIfNotRecording_(new)
    def CudaLibCall(tokens):
        #TODO scan for cudaMemcpy calls
        nonlocal currentNode
        nonlocal translationEnabled
        nonlocal currentLineno
        nonlocal currentLines
        nonlocal keepRecording
        logDetection_("CUDA API call")
        cudaApi, args, finishesOnFirstLine = tokens 
        if not type(currentNode) in [STCudaLibCall,STCudaKernelCall]:
            new = STCudaLibCall(parent=currentNode,lineno=currentLineno,lines=currentLines) 
            new._ignoreInS2STranslation = not translationEnabled
            new.cudaApi  = cudaApi
            #print("finishesOnFirstLine={}".format(finishesOnFirstLine))
            assert type(new._parent) in [STModule,STProcedure,STProgram], type(new._parent)
            new._lines = currentLines
            appendIfNotRecording_(new)
    def CudaKernelCall(tokens):
        nonlocal currentNode
        nonlocal translationEnabled
        nonlocal currentLines
        nonlocal keepRecording
        logDetection_("CUDA kernel call")
        kernelName, kernelLaunchArgs, args, finishesOnFirstLine = tokens 
        assert type(currentNode) in [STModule,STProcedure,STProgram], "type is: "+str(type(currentNode))
        new = STCudaKernelCall(parent=currentNode,lineno=currentLineno,lines=currentLines)
        new._ignoreInS2STranslation = not translationEnabled
        new._lines = currentLines
        appendIfNotRecording_(new)
    def AccDirective(tokens):
        nonlocal currentNode
        nonlocal translationEnabled
        nonlocal currentLines
        nonlocal currentLineno
        nonlocal singleLineStatement
        nonlocal keepRecording
        nonlocal doLoopCtr
        nonlocal directiveNo
        logDetection_("OpenACC directive")
        new = STAccDirective(currentNode,currentLineno,currentLines,directiveNo)
        new._ignoreInS2STranslation = not translationEnabled
        directiveNo = directiveNo + 1
        #msg = "scanner: {}: found acc construct:\t'{}'".format(currentLineno,singleLineStatement)
        #utils.logging.logDebug2(LOG_PREFIX,"AccDirective",msg)
        # if end directive ascend
        if new.isEndDirective() and\
           type(currentNode) is STAccDirective and currentNode.isKernelsDirective():
            ascend_()
            currentNode.append(new)
        # descend in constructs or new node
        elif new.isParallelLoopDirective() or new.isKernelsLoopDirective() or\
             (not new.isEndDirective() and new.isParallelDirective()):
            new = STAccLoopKernel(currentNode,currentLineno,[],directiveNo)
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
    def CufLoopKernel(tokens):
        nonlocal doLoopCtr
        nonlocal currentNode
        nonlocal translationEnabled
        nonlocal currentLineno
        nonlocal currentLines
        nonlocal keepRecording
        nonlocal directiveNo
        logDetection_("CUDA Fortran loop kernel directive")
        new = STCufLoopKernel(currentNode,currentLineno,[],directiveNo)
        new._ignoreInS2STranslation = not translationEnabled
        new._doLoopCtrMemorised=doLoopCtr
        directiveNo += 1
        descend_(new) 
        keepRecording = True
    def Assignment(tokens):
        nonlocal currentNode
        nonlocal translationEnabled
        nonlocal currentLines
        nonlocal currentLineno
        nonlocal singleLineStatement
        logDetection_("assignment")
        if inKernelsAccRegionAndNotRecording():
            parseResult = translator.assignmentBegin.parseString(singleLineStatement)
            lvalue = translator.findFirst(parseResult,translator.TTLValue)
            if not lvalue is None and lvalue.hasMatrixRangeArgs():
                new  = STAccLoopKernel(parent=currentNode,lineno=currentLineno,lines=currentLines)
                new._ignoreInS2STranslation = not translationEnabled
                appendIfNotRecording_(new)
    def GpufortControl(tokens):
        nonlocal singleLineStatement
        nonlocal translationEnabled
        logDetection_("gpufortran control statement")
        if "on" in singleLineStatement:
            translationEnabled = True
        elif "off" in singleLineStatement:
            translationEnabled = False
    
    # TODO completely remove / comment out !$acc end kernels
    moduleStart.setParseAction(Module_visit)
    programStart.setParseAction(Program_visit)
    functionStart.setParseAction(Function_visit)
    subroutineStart.setParseAction(Subroutine_visit)
    structureEnd.setParseAction(Structure_leave)
    
    DO.setParseAction(DoLoop_visit)
    ENDDO.setParseAction(DoLoop_leave)
 
    use.setParseAction(UseStatement)
    CONTAINS.setParseAction(PlaceHolder)
    IMPLICIT.setParseAction(PlaceHolder)
    
    declaration.setParseAction(Declaration)
    attributes.setParseAction(Attributes)
    ALLOCATED.setParseAction(Allocated)
    ALLOCATE.setParseAction(Allocate)
    DEALLOCATE.setParseAction(Deallocate)
    memcpy.setParseAction(Memcpy)
    #pointerAssignment.setParseAction(PointerAssignment)
    nonZeroCheck.setParseAction(NonZeroCheck)

    # CUDA Fortran 
    cuf_kernel_do.setParseAction(CufLoopKernel)
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
        nonlocal currentLines
        nonlocal currentLineno
        nonlocal singleLineStatement

        matched = len(expression.searchString(singleLineStatement,1))
        if matched:
           utils.logging.logDebug3(LOG_PREFIX,"parseFile.scanString","FOUND expression '{}' in line {}: '{}'".format(expressionName,currentLineno,currentLines[0].rstrip()))
        else:
           utils.logging.logDebug4(LOG_PREFIX,"parseFile.scanString","did not find expression '{}' in line {}: '{}'".format(expressionName,currentLineno,currentLines[0].rstrip()))
        return matched
    
    def tryToParseString(expressionName,expression):
        """
        These expressions might never be hidden behind a single-line if or might
        never be an argument of another calls.
        No need to check if comments are in the line as the first
        words of the line must match the expression.
        """
        nonlocal currentLines
        nonlocal currentLineno
        nonlocal singleLineStatement
        
        try:
           expression.parseString(singleLineStatement)
           utils.logging.logDebug3(LOG_PREFIX,"parseFile.tryToParseString","FOUND expression '{}' in line {}: '{}'".format(expressionName,currentLineno,currentLines[0].rstrip()))
           return True
        except ParseBaseException as e: 
           utils.logging.logDebug4(LOG_PREFIX,"parseFile.tryToParseString","did not find expression '{}' in line '{}'".format(expressionName,currentLines[0]))
           utils.logging.logDebug5(LOG_PREFIX,"parseFile.tryToParseString",str(e))
           return False

    for record in records:
        for singleLineStatement in record["expandedStatements"]: 
            currentLineno = record["lineno"]
            currentLines  = record["lines"]
            # host code
            if "cuf" in SOURCE_DIALECTS:
                scanString("attributes",attributes)
                scanString("cudaLibCall",cudaLibCall)
                scanString("cudaKernelCall",cudaKernelCall)
     
            # scan for more complex expressions first      
            scanString("assignmentBegin",assignmentBegin)
            scanString("memcpy",memcpy)
            scanString("allocated",ALLOCATED)
            scanString("deallocate",DEALLOCATE) 
            scanString("allocate",ALLOCATE) 
            scanString("nonZeroCheck",nonZeroCheck)
            #scanString("cpp_ifdef",cpp_ifdef)
            #scanString("cpp_defined",cpp_defined)
            
            # host/device environments  
            #tryToParseString("callEnd",callEnd)
            if not tryToParseString("structureEnd|ENDDO|gpufort_control",structureEnd|ENDDO|gpufort_control):
                tryToParseString("use|CONTAINS|IMPLICIT|declaration|ACC_START|cuf_kernel_do|DO|moduleStart|programStart|functionStart|subroutineStart",\
                   use|CONTAINS|IMPLICIT|declaration|ACC_START|cuf_kernel_do|DO|moduleStart|programStart|functionStart|subroutineStart)
            if keepRecording:
               try:
                  currentNode._lines += currentLines
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
              kernel._lineno in kernelsToConvertToHip or\
              kernel.kernelName() in kernelsToConvertToHip:
                stnode = kernel._parent.findFirst(filter=lambda child: not child.considerInS2STranslation() and type(child) in [STUseStatement,STDeclaration,STPlaceHolder])
                assert not stnode is None
                indent = " "*(len(stnode.lines()[0]) - len(stnode.lines()[0].lstrip()))
                stnode._preamble.add("{0}use {1}\n".format(indent,hipModuleName))
   
    if "cuf" in SOURCE_DIALECTS:
         __postprocessCuf(stree,hipModuleName)
    if "acc" in SOURCE_DIALECTS:
         __postprocessAcc(stree,hipModuleName)
    utils.logging.logLeaveFunction(LOG_PREFIX,"postprocess")
