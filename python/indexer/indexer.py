# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
#!/usr/bin/env python3
import addtoplevelpath
import os,sys
import re
import subprocess
import logging

import orjson

import threading
import concurrent.futures

from multiprocessing import Pool

import translator.translator as translator
import utils

GPUFORT_MODULE_FILE_SUFFIX=".gpufort_mod"

CASELESS    = False
GRAMMAR_DIR = os.path.join(os.path.dirname(__file__),"../grammar")
exec(open("{0}/grammar.py".format(GRAMMAR_DIR)).read())

# configurable parameters
indexerDir = os.path.dirname(__file__)
exec(open("{0}/indexer_options.py.in".format(indexerDir)).read())
    
pFilter        = re.compile(FILTER) 
pAntiFilter    = re.compile(ANTIFILTER)
pContinuation  = re.compile(CONTINUATION_FILTER)
pPrescanFilter = re.compile(PRESCAN_FILTER) 

def __readFortranFile(filepath,gfortranOptions):
    """
    Read and preprocess a Fortran file. Make all
    statements take a single line, i.e. remove all occurences
    of "&".
    """
    def considerLine(strippedLine):
        return (pFilter.match(strippedLine) != None) and (pAntiFilter.match(strippedLine) is None)
    try:
       command = PREPROCESS_FORTRAN_FILE.format(file=filepath,options=gfortranOptions)
       output  = subprocess.check_output(command,shell=True).decode("UTF-8")
       # remove Fortran line continuation and directive continuation
       output = pContinuation.sub(" ",output.lower()) 
       
       # filter statements
       filteredLines = []
       for line in output.split("\n"):
           strippedLine = line.strip().rstrip("\n")
           if considerLine(strippedLine):
               filteredLines.append(strippedLine)
    except subprocess.CalledProcessError as cpe:
        raise cpe
    return filteredLines

class __Node():
    def __init__(self,kind,name,data,parent=None):
        self._kind     = kind
        self._name     = name
        self._parent   = parent 
        self._data     = data
    def __str__(self):
        return "{}: {}".format(self._name,self._data)
    __repr__ = __str__

def __parseFile(fileLines,filepath):
    # Regex
    datatype_reg = Regex(r"\b(type\s*\(|character|integer|logical|real|complex|double\s+precision)\b")

    index = []

    # Currently, we are only interested in a modules declarations
    # and its derived types.
    # Later, we might also parse routines

    accessLock   = threading.Lock()
    taskExecutor = concurrent.futures.ThreadPoolExecutor()
    # statistics
    totalNumTasks = 0
    
    def ParseDeclarationTask_(parentNode,inputText):
        """
        :note: the term 'task' should highlight that this is a function
        that is directly submitted to a thread in the worker pool.
        """
        nonlocal accessLock
        variables =\
          translator.createIndexRecordsFromDeclaration(\
            translator.declaration.parseString(inputText)[0])
        accessLock.acquire()
        parentNode._data["variables"] += variables
        accessLock.release()
    
    postParsingJobs = [] # jobs to run after the file was parsed statement by statement
    class ParseAttributesJob_:
        """
        :note: the term 'job' should highlight that an object of this class
        is put into a list that is submitted to a worker thread pool at the end of the parsing.
        """
        def __init__(self,parentNode,inputText):
            self._parentNode = parentNode
            self._inputText  = inputText
        def run(self):
            nonlocal accessLock
            attribute, modifiedVars = \
                translator.parseAttributes(translator.attributes.parseString(inputText)[0])
            for varContext in self._parentNode._data["variables"]:
                if varContext["name"] in modifiedVars and attribute in varContext:
                    accessLock.acquire()
                    varContext[attribute] = True
                    accessLock.release()
    class ParseAccDeclareJob_:
        """
        :note: the term 'job' should highlight that an object of this class
        is put into a list that is submitted to a worker thread pool at the end of the parsing.
        """
        def __init__(self,parentNode,inputText):
            self._parentNode = parentNode
            self._inputText  = inputText
        def run(self):
            nonlocal accessLock
            parseResult = translator.acc_declare.parseString(inputText)[0]
            for varContext in currentNode._data["variables"]:
                for varName in parseResult.mapAllocVariables():
                    if varContext["name"] == varName:
                        accessLock.acquire()
                        varContext["declareOnTarget"] = "alloc"
                        accesslock.release()
                for varName in parseResult.mapToVariables():
                    if varContext["name"] == varName: 
                        accesslock.acquire()
                        varContext["declareOnTarget"] = "to"
                        accessLock.release()
                for varName in parseResult.mapFromVariables():
                    if varContext["name"] == varName: 
                        accessLock.acquire()
                        varContext["declareOnTarget"] = "from"
                        accessLock.release()
                for varName in parseResult.mapTofromVariables():
                    if varContext["name"] == varName: 
                        accessLock.acquire()
                        varContext["declareOnTarget"] = "tofrom"
                        accessLock.release()

    # Parser events
    root        = __Node("root","root",data=index,parent=None)
    currentNode = root
    currentLine = None

    def createBaseEntry_(kind,name,filepath):
        entry = {}
        entry["kind"]        = kind
        entry["name"]        = name
        #entry["file"]        = filepath
        entry["variables"]   = []
        entry["subprograms"] = []
        entry["usedModules"] = []
        return entry
    def logEnterNode_():
        nonlocal currentNode
        nonlocal currentLine
        utils.logDebug("indexer:\tenter {0} '{1}' in line: '{2}'".format(\
          currentNode._data["kind"],currentNode._data["name"],\
          currentLine))
    def logLeaveNode_():
        nonlocal currentNode
        nonlocal currentLine
        utils.logDebug("indexer:\tleave {0} '{1}' in line: '{2}'".format(\
          currentNode._data["kind"],currentNode._data["name"],\
          currentLine))
    def logDetection_(kind):
        nonlocal currentNode
        nonlocal currentLine
        utils.logDebug2("indexer:\t[current-node={}:{}] FOUND {} in line: '{}'".format(\
                currentNode._kind,currentNode._name,kind,currentLine))
   
    # direct parsing
    def End(tokens):
        nonlocal root
        nonlocal currentNode
        nonlocal currentLine
        logDetection_("end of program/module/subroutine/function")
        if currentNode._kind != "root":
            logLeaveNode_()
            currentNode = currentNode._parent
    def ModuleStart(tokens):
        nonlocal root
        nonlocal currentNode
        name = tokens[0]
        module = createBaseEntry_("module",name,filepath)
        module["types"] = []
        assert currentNode == root
        currentNode._data.append(module)
        currentNode = __Node("module",name,data=module,parent=currentNode)
        logEnterNode_()
    def ProgramStart(tokens):
        nonlocal root
        nonlocal currentNode
        name    = tokens[0]
        program = createBaseEntry_("program",name,filepath)
        program["types"] = []
        assert currentNode._kind == "root"
        currentNode._data.append(program)
        currentNode = __Node("program",name,data=program,parent=currentNode)
        logEnterNode_()
    #host|device,name,[args]
    def SubroutineStart(tokens):
        nonlocal currentNode
        logDetection_("start of subroutine")
        if currentNode._kind != "root":
            name = tokens[1]
            subroutine = createBaseEntry_("subroutine",name,filepath)
            subroutine["attributes"]  = [q.lower() for q in tokens[0]]
            subroutine["dummyArgs"]   = list(tokens[2])
            currentNode._data["subprograms"].append(subroutine)
            currentNode = __Node("subroutine",name,data=subroutine,parent=currentNode)
            logEnterNode_()
    #host|device,name,[args],result
    def FunctionStart(tokens):
        nonlocal currentNode
        logDetection_("start of function")
        if currentNode._kind != "root":
            name = tokens[1]
            function = createBaseEntry_("function",name,filepath)
            function["attributes"]  = [q.lower() for q in tokens[0]]
            function["dummyArgs"]   = list(tokens[2])
            function["resultName"]  = name if tokens[3] is None else tokens[3]
            currentNode._data["subprograms"].append(function)
            currentNode = __Node("function",name,data=function,parent=currentNode)
            logEnterNode_()
    def TypeStart(tokens):
        nonlocal currentNode
        print(currentNode._kind)
        logDetection_("start of type")
        if currentNode._kind != "root":
            assert len(tokens) == 2
            name = tokens[1]
            derivedType = {}
            derivedType["name"]      = name
            derivedType["kind"]      = "type"
            derivedType["variables"] = []
            currentNode._data["types"].append(derivedType)
            currentNode = __Node("type",name,data=derivedType,parent=currentNode)
            logEnterNode_()
    def Use(tokens):
        nonlocal currentNode
        logDetection_("use statement")
        if currentNode._kind != "root":
            usedModule = {}
            usedModule["name"] = translator.makeFStr(tokens[1])
            usedModule["only"] = []
            for pair in tokens[2]:
                original = translator.makeFStr(pair[0])
                renamed = original if pair[1] is None else translator.makeFStr(pair[1])
                usedModule["only"].append({ "original": original, "renamed": renamed })
            #print(currentNode)
            currentNode._data["usedModules"].append(usedModule) # TODO only include what is necessary
    
    # delayed parsing
    
    def Declaration(tokens):
        nonlocal root
        nonlocal currentNode
        nonlocal currentLine
        nonlocal taskExecutor
        nonlocal totalNumTasks
        #print(currentLine)
        logDetection_("declaration")
        if currentNode != root:
            totalNumTasks += 1
            taskExecutor.submit(ParseDeclarationTask_,currentNode,currentLine) 
    def Attributes(s,loc,tokens):
        """
        Add attributes to previously declared variables in same scope/declaration list.
        Does not modify scope of other variables.
        """
        nonlocal root
        nonlocal currentNode
        nonlocal currentLine
        #print(currentLine)
        logDetection_("attributes statement")
        if currentNode != root:
            job = ParseAttributesJob_(currentNode,currentLine) 
            postParsingJobs.append(job)
    def AccDeclare(s,loc,tokens):
        """
        Add attributes to previously declared variables in same scope.
        Does not modify scope of other variables.
        """
        # TODO investigate if target of attribute must be in same scope or not!
        nonlocal root
        nonlocal currentNode
        nonlocal currentLine
        logDetection_("acc declare directive")
        if currentNode != root:
            job = ParseAccDeclareJob_(currentNode,currentLine) 
            postParsingJobs.append(job)
            # 'use kinds, only: dp, sp => sp2' --> [None, 'kinds', [['dp', None], ['sp', 'sp2']]]
    
    moduleStart.setParseAction(ModuleStart)
    typeStart.setParseAction(TypeStart)
    programStart.setParseAction(ProgramStart)
    functionStart.setParseAction(FunctionStart)
    subroutineStart.setParseAction(SubroutineStart)
    acc_declare.setParseAction(AccDeclare)

    typeEnd.setParseAction(End)
    structureEnd.setParseAction(End)

    datatype_reg.setParseAction(Declaration)
    use.setParseAction(Use)
    attributesLhs.setParseAction(Attributes)
    # TODO openacc pragmas

    def tryToParseString(expressionName,expression):
        try:
           expression.parseString(currentLine)
           return True
        except ParseBaseException as e: 
           utils.logDebug3("indexer:\tdid not find expression '{}' in line '{}'".format(expressionName,currentLine))
           utils.logDebug4(str(e))
           return False

    for currentLine in fileLines:
        utils.logDebug3("indexer:\tprocessing line '{}'".format(currentLine))
        # typeStart must be tried before datatype_reg
        tryToParseString("structureEnd|typeEnd|typeStart|declaration|use|attributes|acc_declare|moduleStart|programStart|functionStart|subroutineStart",\
          typeEnd|structureEnd|typeStart|datatype_reg|use|attributesLhs|acc_declare|moduleStart|programStart|functionStart|subroutineStart)
    taskExecutor.shutdown(wait=True) # waits till all tasks have been completed

    # apply attributes and acc variable modifications
    with concurrent.futures.ThreadPoolExecutor() as jobExecutor:
        for job in postParsingJobs:
            jobExecutor.submit(job.run)
    postParsingJobs.clear()

    return index

def __writeJsonFile(index,filepath):
    global PRETTY_PRINT_INDEX_FILE
    with open(filepath,"wb") as outfile:
         if PRETTY_PRINT_INDEX_FILE:
             outfile.write(orjson.dumps(index,option=orjson.OPT_INDENT_2))
         else:
             outfile.write(orjson.dumps(index))

def __readJsonFile(filepath):
    try:
       with open(filepath,"rb") as infile:
            return orjson.loads(infile.read())
    except Exception as e:
        raise e

# API
def scanFile(filepath,gfortranOptions,index):
    """
    Creates an index from a single file.
    """
    filteredLines = __readFortranFile(filepath,gfortranOptions)
    index += __parseFile(filteredLines,filepath)

def writeGpufortModuleFiles(index,outputDir):
    """
    Per module / program found in the index
    write a GPUFORT module file.
    
    :param list index:    [in] Empty or non-empty list.
    :param str outputDir: [in] Output directory.
    """
    for mod in index:
        filepath = outputDir + "/" + mod["name"] + GPUFORT_MODULE_FILE_SUFFIX
        __writeJsonFile(mod,filepath)

def loadGpufortModuleFiles(inputDirs,index):
    """
    Load gpufort module files and append to the index.

    :param list inputDirs: [in] List of input directories (as strings).
    :param list index:     [inout] Empty or non-empty list. Loaded data structure is appended.
    """
    for inputDir in inputDirs:
         for child in os.listdir(inputDir):
             if child.endswith(GPUFORT_MODULE_FILE_SUFFIX):
                 moduleAlreadyExists = False
                 for mod in index:
                     if mod == child.replace(GPUFORT_MODULE_FILE_SUFFIX,""):
                         moduleAlreadyExists = True
                         break
                 if not moduleAlreadyExists:
                     modIndex = __readJsonFile(os.path.join(inputDir, child))
                     index.append(modIndex)

