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
import indexer.indexertools as indexertools
import pyparsingtools

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
        logging.getLogger("").error(msg)
        sys.exit(SCANNER_ERROR_CODE)

def handleIncludeStatements(fortranFilePath,lines):
    """
    Copy included files' content into current file.
    """
    def processLine(line):
        if "#include" in line.lower():
            relativeSnippetPath = line.split(' ')[-1].replace('\"','').replace('\'','').strip()
            snippetPath = os.path.dirname(fortranFilePath) + "/" + relativeSnippetPath
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
    return result

# Pyparsing Actions that create scanner tree (ST)
def parseFile(fortranFilePath):
    """
    Generate an object tree (OT). 
    """
    translationEnabled = TRANSLATION_ENABLED_BY_DEFAULT
    
    current=STRoot()
    doLoopCtr = 0     
    keepRecording = False
    currentFile = str(fortranFilePath)
    currentLineno = -1
    currentLines  = []
    directiveNo   = 0

    def descend(new):
        nonlocal current
        current.append(new)
        current=new
    def ascend():
        nonlocal current
        nonlocal currentFile
        assert not current._parent is None, "In file {}: parent of {} is none".format(currentFile,type(current))
        current = current._parent
   
    # parse actions
    def Module_visit(tokens):
        nonlocal current
        nonlocal translationEnabled
        nonlocal currentLineno
        nonlocal currentLines
        new = STModule(tokens[0],current,currentLineno)
        new._ignoreInS2STranslation = not translationEnabled
        descend(new)
    def Program_visit(tokens):
        nonlocal current
        nonlocal translationEnabled
        nonlocal currentLineno
        nonlocal currentLines
        new = STProgram(tokens[0],current,currentLineno)
        new._ignoreInS2STranslation = not translationEnabled
        descend(new)
    def Function_visit(tokens):
        nonlocal current
        nonlocal translationEnabled
        nonlocal keepRecording
        new = STFunction(qualifier=tokens[0],name=tokens[1],dummyArgs=tokens[2],\
                parent=current,lineno=currentLineno,lines=currentLines)
        new._ignoreInS2STranslation = not translationEnabled
        if new.qualifier.lower() in ["global","device","host,device"]:
            keepRecording = True
        descend(new)
    def Subroutine_visit(tokens):
        nonlocal current
        nonlocal translationEnabled
        nonlocal keepRecording
        new = STSubroutine(qualifier=tokens[0],name=tokens[1],dummyArgs=tokens[2],\
                parent=current,lineno=currentLineno,lines=currentLines)
        new._ignoreInS2STranslation = not translationEnabled
        if new._qualifier.lower() in ["global","device"]:
            keepRecording = True
        descend(new)
    def Structure_leave(tokens):
        nonlocal current
        nonlocal keepRecording
        assert type(current) in [STModule,STProgram,STSubroutine,STFunction], "In file {}: line {}: type is {}".format(currentFile, currentLineno, type(current))
        if type(current) in [STSubroutine,STFunction] and current._qualifier.lower() in ["global","device"]:
            current._lines += currentLines
            keepRecording = False
        ascend()
    def inKernelsAccRegionAndNotRecording():
        nonlocal current
        nonlocal keepRecording
        return not keepRecording and\
            (type(current) is STAccDirective) and\
            (current.isKernelsDirective())
    def DoLoop_visit(tokens):
        nonlocal keepRecording
        nonlocal translationEnabled
        nonlocal doLoopCtr
        if inKernelsAccRegionAndNotRecording():
            new = STAccLoopKernel(parent=current,lineno=currentLineno,lines=currentLines)
            new._ignoreInS2STranslation = not translationEnabled
            new._lines = []
            new._doLoopCtrMemorised=doLoopCtr
            descend(new) 
            keepRecording = True
        doLoopCtr += 1
    def DoLoop_leave(tokens):
        nonlocal current
        nonlocal currentLines
        nonlocal doLoopCtr
        nonlocal keepRecording
        doLoopCtr -= 1
        if isinstance(current, STLoopKernel):
            if keepRecording and current._doLoopCtrMemorised == doLoopCtr:
                current._lines += currentLines
                ascend()
                keepRecording = False
    def Declaration(tokens):
        nonlocal current
        nonlocal translationEnabled
        nonlocal currentLineno
        nonlocal currentLines
        new = STDeclaration(parent=current,lineno=currentLineno,lines=currentLines)
        new._ignoreInS2STranslation = not translationEnabled
        current.append(new)
    def Attributes(tokens):
        nonlocal current
        nonlocal translationEnabled
        nonlocal currentLineno
        nonlocal currentLines
        new = STAttributes(parent=current,lineno=currentLineno,lines=currentLines)
        new._ignoreInS2STranslation = not translationEnabled
        current.append(new)
    def UseStatement(tokens):
        nonlocal current
        nonlocal currentLineno
        nonlocal currentLines
        new = STUseStatement(parent=current,lineno=currentLineno,lines=currentLines)
        new._ignoreInS2STranslation = not translationEnabled
        new._name = translator.makeFStr(tokens[1]) # just get the name, ignore specific includes
        current.append(new)
    def PlaceHolder(tokens):
        nonlocal current
        nonlocal translationEnabled
        nonlocal currentLineno
        nonlocal currentLines
        new = STPlaceHolder(current,currentLineno,currentLines)
        new._ignoreInS2STranslation = not translationEnabled
        current.append(new)
    def NonZeroCheck(tokens):
        nonlocal current
        nonlocal translationEnabled
        nonlocal currentLineno
        nonlocal currentLines
        new=STNonZeroCheck(parent=current,lineno=currentLineno,lines=currentLines)
        new._ignoreInS2STranslation = not translationEnabled
        current.append(new)
    def Allocated(tokens):
        nonlocal current
        nonlocal translationEnabled
        nonlocal currentLineno
        nonlocal currentLines
        new=STAllocated(parent=current,lineno=currentLineno,lines=currentLines)
        new._ignoreInS2STranslation = not translationEnabled
        current.append(new)
    def Allocate(tokens):
        nonlocal current
        nonlocal translationEnabled
        nonlocal currentLineno
        nonlocal currentLines
        new=STAllocate(parent=current,lineno=currentLineno,lines=currentLines)
        new._ignoreInS2STranslation = not translationEnabled
        current.append(new)
    def Deallocate(tokens):
        nonlocal current
        nonlocal translationEnabled
        nonlocal currentLineno
        nonlocal currentLines
        # TODO filter variable, replace with hipFree
        new = STDeallocate(parent=current,lineno=currentLineno,lines=currentLines)
        new._ignoreInS2STranslation = not translationEnabled
        current.append(new)
    def Memcpy(tokens):
        nonlocal current
        nonlocal translationEnabled
        nonlocal currentLineno
        nonlocal currentLines
        new = STMemcpy(parent=current,lineno=currentLineno,lines=currentLines)
        new._ignoreInS2STranslation = not translationEnabled
        current.append(new)
    def CudaLibCall(tokens):
        #TODO scan for cudaMemcpy calls
        nonlocal current
        nonlocal translationEnabled
        nonlocal currentLineno
        nonlocal currentLines
        nonlocal keepRecording
        cudaApi, args, finishesOnFirstLine = tokens 
        if not type(current) in [STCudaLibCall,STCudaKernelCall]:
            new = STCudaLibCall(parent=current,lineno=currentLineno,lines=currentLines) 
            new._ignoreInS2STranslation = not translationEnabled
            new.cudaApi  = cudaApi
            #print("finishesOnFirstLine={}".format(finishesOnFirstLine))
            assert type(new._parent) in [STModule,STFunction,STSubroutine,STProgram], type(new._parent)
            new._lines = currentLines
            current.append(new)
    def CudaKernelCall(tokens):
        nonlocal current
        nonlocal translationEnabled
        nonlocal currentLines
        nonlocal keepRecording
        kernelName, kernelLaunchArgs, args, finishesOnFirstLine = tokens 
        assert type(current) in [STModule,STFunction,STSubroutine,STProgram], "type is: "+str(type(current))
        new = STCudaKernelCall(parent=current,lineno=currentLineno,lines=currentLines)
        new._ignoreInS2STranslation = not translationEnabled
        new._lines = currentLines
        current.append(new)
    def AccDirective(tokens):
        nonlocal current
        nonlocal translationEnabled
        nonlocal currentLines
        nonlocal currentLineno
        nonlocal singleLineStatement
        nonlocal keepRecording
        nonlocal doLoopCtr
        nonlocal directiveNo
        new = STAccDirective(current,currentLineno,currentLines,directiveNo)
        new._ignoreInS2STranslation = not translationEnabled
        directiveNo = directiveNo + 1
        msg = "scanner: {}: found acc construct:\t'{}'".format(currentLineno,singleLineStatement)
        logging.getLogger("").debug(msg) ; # print(msg)
        # if end directive ascend
        if new.isEndDirective() and\
           type(current) is STAccDirective and current.isKernelsDirective():
            ascend()
            current.append(new)
        # descend in constructs or new node
        elif new.isParallelLoopDirective() or new.isKernelsLoopDirective() or\
             (not new.isEndDirective() and new.isParallelDirective()):
            new = STAccLoopKernel(current,currentLineno,[],directiveNo)
            new._ignoreInS2STranslation = not translationEnabled
            new._doLoopCtrMemorised=doLoopCtr
            descend(new)  # descend also appends 
            keepRecording = True
        elif not new.isEndDirective() and new.isKernelsDirective():
            descend(new)  # descend also appends 
            #print("descending into kernels or parallel construct")
        else:
           # append new directive
           current.append(new)
    def CufLoopKernel(tokens):
        nonlocal doLoopCtr
        nonlocal current
        nonlocal translationEnabled
        nonlocal currentLineno
        nonlocal currentLines
        nonlocal keepRecording
        nonlocal directiveNo
        new = STCufLoopKernel(current,currentLineno,[],directiveNo)
        new._ignoreInS2STranslation = not translationEnabled
        new._doLoopCtrMemorised=doLoopCtr
        directiveNo += 1
        descend(new) 
        keepRecording = True
    def Assignment(tokens):
        nonlocal current
        nonlocal translationEnabled
        nonlocal currentLines
        nonlocal currentLineno
        nonlocal singleLineStatement
        if inKernelsAccRegionAndNotRecording():
            parseResult = translator.assignmentBegin.parseString(singleLineStatement)
            lvalue = translator.findFirst(parseResult,translator.TTLValue)
            if not lvalue is None and lvalue.hasMatrixRangeArgs():
                new  = STAccLoopKernel(parent=current,lineno=currentLineno,lines=currentLines)
                new._ignoreInS2STranslation = not translationEnabled
                current.append(new)
    def GpufortControl(tokens):
        nonlocal singleLineStatement
        nonlocal translationEnabled
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

    currentFile = str(fortranFilePath)
    current._children.clear()
    def scanString(expressionName,expression):
        """
        These expressions might be hidden behind a single-line if.
        """
        nonlocal currentLines
        nonlocal currentLineno
        nonlocal singleLineStatement

        matched = len(expression.searchString(singleLineStatement,1))
        if matched:
           logging.getLogger("").debug("scanner::scanString\tFOUND expression '{}' in line {}: '{}'".format(expressionName,currentLineno,currentLines[0].rstrip()))
        else:
           logging.getLogger("").debug2("scanner::scanString\tdid not find expression '{}' in line {}: '{}'".format(expressionName,currentLineno,currentLines[0].rstrip()))
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
           logging.getLogger("").debug("scanner::tryToParseString\tFOUND expression '{}' in line {}: '{}'".format(expressionName,currentLineno,currentLines[0].rstrip()))
           return True
        except ParseBaseException as e: 
           logging.getLogger("").debug2("scanner::tryToParseString\tdid not find expression '{}' in line '{}'".format(expressionName,currentLines[0]))
           logging.getLogger("").debug3(str(e))
           return False
    
    pDirectiveContinuation = re.compile(r"\n[!c\*]\$\w+\&")
    pContinuation          = re.compile(r"(\&\s*\n)|(\n[!c\*]\$\w+\&)")
    with open(fortranFilePath,"r") as fortranFile:
        # 1. Handle all include statements
        lines = handleIncludeStatements(fortranFilePath,fortranFile.readlines())
        # 2. collapse multi-line statements (&)
        buffering = False
        lineStarts = []
        for lineno,line in enumerate(lines):
            # Continue buffering if multiline CUF/ACC/OMP statement
            buffering |= pDirectiveContinuation.match(line) != None
            if not buffering:
                lineStarts.append(lineno)
            if line.rstrip().endswith("&"):
                buffering = True
            else:
                buffering = False
        lineStarts.append(len(lines))
        # TODO merge this with loop above
        # 3. now go through the collapsed lines
        for i,_ in enumerate(lineStarts[:-1]):
            lineStart     = lineStarts[i]
            lineEnd       = lineStarts[i+1]
            currentLineno = lineStart
            currentLines  = lines[lineStart:lineEnd]
            
            # make lower case, replace line continuation by whitespace, split at ";"
            preprocessedLines = pContinuation.sub(" "," ".join(currentLines).lower()).split(";")
            for singleLineStatement in preprocessedLines: 
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
                      current._lines += currentLines
                   except Exception as e:
                      logging.getLogger("").error("While parsing file {}".format(currentFile))
                      raise e
    assert type(current) is STRoot
    return current

def postprocessAcc(stree,hipModuleName):
    """
    Add use statements as well as handles plus their creation and destruction for certain
    math libraries.
    """
    global DESTINATION_DIALECT
    global RUNTIME_MODULE_NAMES
    
    # acc detection
    directives = stree.findAll(filter=lambda node: isinstance(node,STAccDirective), recursively=True)
    for directive in directives:
         stnode = directive._parent.findFirst(filter=lambda child : type(child) in [STUseStatement,STDeclaration,STPlaceHolder])
         if not stnode is None:
             indent = " "*(len(stnode.lines()[0]) - len(stnode.lines()[0].lstrip()))
             accRuntimeModuleName = RUNTIME_MODULE_NAMES[DESTINATION_DIALECT]
             if accRuntimeModuleName != None and len(accRuntimeModuleName):
                 stnode._preamble.add("{0}use iso_c_binding\n{0}use {1}\n".format(indent,accRuntimeModuleName))
    
def postprocessCuf(stree,hipModuleName):
    """
    Add use statements as well as handles plus their creation and destruction for certain
    math libraries.
    """
    global CUBLAS_VERSION 
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

def postprocess(stree,hipModuleName):
    """
    Add use statements as well as handles plus their creation and destruction for certain
    math libraries.
    """
    if "hip" in DESTINATION_DIALECT or len(KERNELS_TO_CONVERT_TO_HIP):
        # insert use kernel statements at appropriate point
        def isLoopKernel(child):
            return isinstance(child,STLoopKernel) or\
                   (type(child) is STSubroutine and child.isAcceleratorRoutine())
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
         postprocessCuf(stree,hipModuleName)
    
    if "acc" in SOURCE_DIALECTS:
         postprocessAcc(stree,hipModuleName)
