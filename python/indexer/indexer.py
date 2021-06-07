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

def __discoverInputFiles(searchDir):
    """
    Discover 
    """
    try:
       command = DISCOVER_INPUT_FILES.format(search_dir=searchDir)
       output = subprocess.check_output(command,shell=True).decode('UTF-8')
       result = output.rstrip("\n").split("\n")
       return result
    except subprocess.CalledProcessError as cpe:
       raise cpe

class __Node():
    def __init__(self,name,data,parent=None):
        self._name     = name
        self._parent   = parent 
        self._data     = data
    def __str__(self):
        return "{}: {}".format(self._name,self._data)
    __repr__ = __str__

def __parseFile(fileLines,filepath):
    # Regex
    moduleStart_reg      = Regex(r"\bmodule\b")
    programStart_reg     = Regex(r"\bprogram\b")
    datatype_reg         = Regex(r"\b(type|character|integer|logical|real|complex|double\s+precision)\b")
    use_reg              = Regex(r"\buse\b\s+\w+")

    index = []

    # Currently, we are only interested in a modules declarations
    # and its derived types.
    # Later, we might also parse routines

    root    = __Node("root",data=index,parent=None)
    currentNode = root
    currentLine = None

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
    def createBaseEntry_(typeName,name,tag,filepath):
        entry = {}
        entry["type"]                            = typeName
        entry["name"]                            = name
        entry["tag"]                             = tag
        entry["file"]                            = filepath
        entry["variables"]                       = []
        entry["subprograms"]                     = []
        entry["types"]                           = []
        entry["usedModules"]  = []
        return entry

    def End(tokens):
        nonlocal root
        nonlocal currentNode
        if currentNode != root:
            currentNode = currentNode._parent
    def ModuleStart(tokens):
        nonlocal root
        nonlocal currentNode
        name = tokens[0]
        module = createBaseEntry_("module",name,name,filepath)
        root._data.append(module)
        currentNode = __Node("module",data=module,parent=currentNode)
    def ProgramStart(tokens):
        nonlocal root
        nonlocal currentNode
        name    = tokens[0]
        program = createBaseEntry_("program",name,name,filepath)
        root._data.append(program)
        currentNode = __Node("program",data=program,parent=currentNode)
    #host|device,name,[args]
    def SubroutineStart(tokens):
        nonlocal root
        nonlocal currentNode
        if currentNode != root:
            name = tokens[1]
            subroutine = createBaseEntry_("subroutine",name,name,filepath)
            if currentNode != root:
                subroutine["tag"] = currentNode._data["name"] + ":" + name
            subroutine["attributes"]  = [q.lower() for q in tokens[0]]
            subroutine["dummyArgs"]   = list(tokens[2])
            currentNode._data["subprograms"].append(subroutine)
            currentNode = __Node("subroutine",data=subroutine,parent=currentNode)
    #host|device,name,[args],result
    def FunctionStart(tokens):
        nonlocal root
        nonlocal currentNode
        if currentNode != root:
            name = tokens[1]
            function = createBaseEntry_("function",name,name,filepath)
            if currentNode != root:
                function["tag"] = currentNode._data["name"] + ":" + name
            function["attributes"]  = [q.lower() for q in tokens[0]]
            function["dummyArgs"]   = list(tokens[2])
            function["resultName"]  = name if tokens[3] is None else tokens[3]
            currentNode._data["subprograms"].append(function)
            currentNode = __Node("function",data=function,parent=currentNode)
    def TypeStart(tokens):
        nonlocal root
        nonlocal currentNode
        if currentNode != root:
            name = tokens[1]
            derivedType = {}
            derivedType["name"]      = name
            derivedType["variables"] = []
            currentNode._data["types"].append(derivedType)
    def Declaration(s,loc,tokens):
        nonlocal root
        nonlocal currentNode
        nonlocal currentLine
        nonlocal taskExecutor
        nonlocal totalNumTasks
        #print(currentLine)
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
        parseResult = translator.acc_declare.parseString(currentLine)[0]
        if currentNode != root:
            job = ParseAccDeclareJob_(currentNode,currentLine) 
            postParsingJobs.append(job)
            # 'use kinds, only: dp, sp => sp2' --> [None, 'kinds', [['dp', None], ['sp', 'sp2']]]
    def Use(tokens):
        nonlocal root
        nonlocal currentNode
        if currentNode != root:
            usedModule = {}
            usedModule["name"] = translator.makeFStr(tokens[1])
            usedModule["only"] = {}
            for pair in tokens[2]:
                original = translator.makeFStr(pair[0])
                renaming = original if pair[1] is None else translator.makeFStr(pair[1])
                usedModule["only"][original]=renaming
            #print(currentNode)
            currentNode._data["usedModules"].append(usedModule) # TODO only include what is necessary
    
    moduleStart_reg.setParseAction(ModuleStart)
    typeStart.setParseAction(TypeStart)
    programStart_reg.setParseAction(ProgramStart)
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
           utils.logDebug("indexer:\tFOUND expression '{}' in line: '{}'".format(expressionName,currentLine))
           return True
        except ParseBaseException as e: 
           utils.logDebug("indexer:\tdid not find expression '{}' in line '{}'".format(expressionName,currentLine),debugLevel=2)
           utils.logDebug(str(e),debugLevel=3)
           return False

    for currentLine in fileLines:
        tryToParseString("structureEnd|typeEnd|declaration|use|attributes|acc_declare|typeStart|moduleStart|programStart|functionStart|subroutineStart",\
          typeEnd|structureEnd|datatype_reg|use_reg|attributesLhs|acc_declare|typeStart|moduleStart_reg|programStart_reg|functionStart|subroutineStart)
    taskExecutor.shutdown(wait=True) # waits till all tasks have been completed

    # apply attributes and acc variable modifications
    with concurrent.futures.ThreadPoolExecutor() as jobExecutor:
        for job in postParsingJobs:
            jobExecutor.submit(job.run)
    postParsingJobs.clear()

    return index

def __resolveDependencies_body(i,index):
    def ascend(module):
        """
        """
        nonlocal index
        nonlocal i
        for used in module["usedModules"]:
            name = used["name"]
            only = used["only"]
            usedModule = next((m for m in index if m["name"] == name),None)
            if usedModule != None:
                ascend(usedModule)
                if len(only):
                    variables   = []
                    types       = []
                    subprograms = []
                    for var in usedModule["variables"]:
                        if var["name"] in only:
                            var["name"] = only[var["name"]]
                            variables.append(var)
                    for struct in usedModule["types"]:
                        if struct["name"] in only:
                            struct["name"] = only[struct["name"]]
                            types.append(struct)
                    for subprogram in usedModule["subprograms"]:
                        if subprogram["name"] in only:
                            subprogram["name"] = only[subprogram["name"]]
                            subprograms.append(subprogram)
                else:
                    variables   = usedModule["variables"]
                    types       = usedModule["types"]
                    subprograms = usedModule["subprograms"]
                localVarNames        = [var["name"]  for var in index[i]["variables"]]
                localTypeNames       = [typ["name"]  for typ in index[i]["types"]]
                localSubprogramNames = [prog["name"] for prog in index[i]["subprograms"]]
                index[i]["variables"]   = [var for var in variables     if var["name"] not in localVarNames]  +\
                   index[i]["variables"]
                index[i]["types"]       = [typ for typ in types         if typ["name"] not in localTypeNames] +\
                   index[i]["types"]  
                index[i]["subprograms"] = [prog for prog in subprograms if prog["name"] not in localSubprogramNames] +\
                   index[i]["subprograms"]  
    ascend(index[i])
    # resolve dependencies
    return i,index[i]

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
        __writeJsonFile(index,filepath)

def loadGpufortModuleFiles(inputDirs,index):
    """
    Load gpufort module files and append to the index.

    :param list inputDirs: [in] List of input directories (as strings).
    :param list index:     [inout] Empty or non-empty list. Loaded data structure is appended.
    """
    for inputDir in inputDirs:
         for child in os.listdir(inputDir):
             if child.endswith(GPUFORT_MODULE_FILE_SUFFIX):
                 #if not len(searchedModules) or child.replace(GPUFORT_MODULE_FILE_SUFFIX,"") in searchedModules:
                     modIndex = __readJsonFile(os.path.join(inputDir, child))
                     index.append(modIndex)

# TODO old code, keep here for now as reference

#def dependencyGraphs(index):
#    # discover root nodes (serial)
#    discoveredModuleNames = [module["name"] for module in index]
#    graphs = []
#    for module in index:
#        isNotRoot = False
#        for usedModule in module["usedModules"]:
#            isNotRoot = isNotRoot or usedModule["name"] in discoveredModuleNames
#        if not isNotRoot:
#            graphs.append(__Node(module["name"],data=module))
#    # build tree (parallel)
#    handles = []
#    with Pool(processes=max(1,int(len(graphs)/2))) as pool: # untuned
#         handles = [pool.apply_async(__dependencyGraphs_descend, (root,index,)) for root in graphs]
#         pool.close()
#         pool.join()
#         for i,h in enumerate(handles):
#            graphs[i] = h.get()
#    return graphs
#
#def resolveDependencies(index,searchedFiles=[],searchedTags=[]):
#    def select(module):
#        nonlocal searchedFiles
#        nonlocal searchedTags
#        considerFile = not len(searchedFiles) or module["file"] in searchedFiles
#        considerTag  = not len(searchedTags) or module["tag"] in searchedTags
#        return considerFile and considerTag
#
#    selection = [i for i,module in enumerate(index) if select(module) and len(module["usedModules"])]
# 
#    if len(selection):
#        with Pool(processes=len(selection)) as pool: # untuned
#            handles = [pool.apply_async(__resolveDependencies_body, (i,index,)) for i in selection]
#            pool.close()
#            pool.join()
#            for h in handles:
#                i,result = h.get()
#                index[i] = result
#    # filter out not needed entries 
#    return [module for module in index if select(module)]
#def scanSearchDirs(searchDirs,optionsAsStr):
#    global SCAN_SEARCH_DIRS_MAX_PROCESSES
#
#    index = []
#    inputFiles = []
#    for searchDir in searchDirs:
#        if os.path.exists(searchDir):
#            inputFiles += __discoverInputFiles(searchDir)
#        else:
#            msg = "indexer: include directory '{}' does not exist. Is ignored.".format(searchDir)
#            utils.logWarn(msg)
#    partialResults = []
#    print("\n".join(inputFiles))
#    with Pool(processes=min(SCAN_SEARCH_DIRS_MAX_PROCESSES,len(inputFiles))) as pool: #untuned
#        fileLines = [__readFortranFile(inputFile,optionsAsStr) for i,inputFile in enumerate(inputFiles)]
#        partialResults = [pool.apply_async(__parseFile, (fileLines[i],inputFile,)) for i,inputFile in enumerate(inputFiles)]
#        pool.close()
#        pool.join()
#    for p in partialResults:
#        index += p.get()
#    return index
#
