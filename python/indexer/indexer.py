# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
#!/usr/bin/env python3
import addtoplevelpath
import os,sys
import re
import subprocess
import logging
import json

from multiprocessing import Pool

import translator.translator as translator
import utils

CASELESS    = False
GRAMMAR_DIR = os.path.join(os.path.dirname(__file__),"../grammar")
exec(open("{0}/grammar.py".format(GRAMMAR_DIR)).read())

# configurable parameters
indexerDir = os.path.dirname(__file__)
exec(open("{0}/indexer_options.py.in".format(indexerDir)).read())
    
pContinuation = re.compile(CONTINUATION)
pFilter       = re.compile(FILTER) 
pAntiFilter   = re.compile(ANTIFILTER)

def __readFortranFile(filepath,compilerOptions):
    def considerLine(strippedLine):
        return (pFilter.match(strippedLine) != None) and (pAntiFilter.match(strippedLine) is None)

    """
    Read and preprocess a Fortran file. Make all
    statements take a single line, i.e. remove all occurences
    of "&".
    """
    try:
       command = PREPROCESS_FORTRAN_FILE.format(file=filepath,options=compilerOptions)
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
        self._children = []
        self._data     = data
    def __str__(self):
        if len(self._children):
            return "{}: {}".format(self._name,self._children)
        else:
            return self._name
    __repr__ = __str__

def __parseFile(fileLines,filePath):
    index = []

    # Currently, we are only interested in a modules declarations
    # and its derived types.
    # Later, we might also parse routines

    root    = __Node("root",data=index,parent=None)
    current = root
    currentLine = None

    def createBaseEntry_(typeName,name,tag,filePath):
        entry = {}
        entry["type"]                            = typeName
        entry["name"]                            = name
        entry["tag"]                             = tag
        entry["file"]                            = filePath
        entry["types"]                           = []
        entry["variables"]                       = []
        entry["usedModulesOrParentSubprograms"]  = []
        return entry

    def End(tokens):
        nonlocal current
        current = current._parent
    def ModuleStart(tokens):
        nonlocal current
        name = tokens[0]
        module = createBaseEntry_("module",name,name,filePath)
        root._data.append(module)
        current = __Node("module",data=module,parent=current)
    def ProgramStart(tokens):
        nonlocal current
        name    = tokens[0]
        program = createBaseEntry_("program",name,name,filePath)
        root._data.append(program)
        current = __Node("program",data=program,parent=current)
    #host|device,name,[args]
    def SubroutineStart(tokens):
        nonlocal current
        name = tokens[1]
        subroutine = createBaseEntry_("subroutine",name,name,filePath)
        if current != root:
            subroutine["tag"] = current._data["name"] + ":" + name
        subroutine["attributes"]  = [q.lower() for q in tokens[0]]
        subroutine["dummyArgs"]   = list(tokens[2])
        if current._name not in ["root","module"]:
            subroutine["usedModulesOrParentSubprograms"].append({ "name": current._data["name"], "only": []})
        root._data.append(subroutine)
        current = __Node("subroutine",data=subroutine,parent=current)
    #host|device,name,[args],result
    def FunctionStart(tokens):
        nonlocal current
        name = tokens[1]
        function = createBaseEntry_("function",name,name,filePath)
        if current != root:
            function["tag"] = current._data["name"] + ":" + name
        function["attributes"]  = [q.lower() for q in tokens[0]]
        function["dummyArgs"]   = list(tokens[2])
        function["resultName"]  = name if tokens[3] is None else tokens[3]
        if current._name not in ["root","module"]:
            function["usedModulesOrParentSubprograms"].append({ "name": current._data["name"], "only": []})
        root._data.append(function)
        current = __Node("function",data=function,parent=current)
    
    def TypeStart(tokens):
        nonlocal current
        name = tokens[1]
        derivedType = {}
        derivedType["name"]      = name
        derivedType["variables"] = []
        current._data["types"].append(derivedType)
        current = __Node("type",data=derivedType,parent=current)
    def Declaration(s,loc,tokens):
        nonlocal current
        nonlocal currentLine
        #print(currentLine)
        current._data["variables"] +=\
          translator.createIndexRecordsFromDeclaration(\
            translator.declaration.parseString(currentLine)[0])
    def Attributes(s,loc,tokens):
        """
        Add attributes to previously declared variables in same scope.
        Does not modify scope of other variables.
        """
        # TODO investigate if target of attribute must be in same scope or not!
        nonlocal current
        nonlocal currentLine
        #print(currentLine)
        attribute, modifiedVars = \
            translator.parseAttributes(translator.attributes.parseString(currentLine)[0])
        for varContext in current._data["variables"]:
            if varContext["name"] in modifiedVars and attribute in varContext:
                varContext[attribute] = True
    # 'use kinds, only: dp, sp => sp2' --> [None, 'kinds', [['dp', None], ['sp', 'sp2']]]
    def Use(tokens):
        nonlocal current
        usedModule = {}
        usedModule["name"] = translator.makeFStr(tokens[1])
        usedModule["only"] = {}
        for pair in tokens[2]:
            original = translator.makeFStr(pair[0])
            renaming = original if pair[1] is None else translator.makeFStr(pair[1])
            usedModule["only"][original]=renaming
        current._data["usedModulesOrParentSubprograms"].append(usedModule) # TODO only include what is necessary
    
    moduleStart.setParseAction(ModuleStart)
    typeStart.setParseAction(TypeStart)
    programStart.setParseAction(ProgramStart)
    functionStart.setParseAction(FunctionStart)
    subroutineStart.setParseAction(SubroutineStart)

    typeEnd.setParseAction(End)
    structureEnd.setParseAction(End)

    declarationLhs.setParseAction(Declaration)
    use.setParseAction(Use)
    attributesLhs.setParseAction(Attributes)
    # TODO openacc pragmas

    def tryToParseString(expressionName,expression):
        try:
           expression.parseString(currentLine)
           logging.getLogger("").debug("indexer:\tFOUND expression '{}' in line: '{}'".format(expressionName,currentLine))
           return True
        except ParseBaseException as e: 
           logging.getLogger("").debug2("indexer:\tdid not find expression '{}' in line '{}'".format(expressionName,currentLine))
           logging.getLogger("").debug3(str(e))
           return False

    for currentLine in fileLines:
        tryToParseString("structureEnd|typeEnd|declaration|use|attributes|typeStart|moduleStart|programStart|functionStart|subroutineStart",\
            typeEnd|structureEnd|declarationLhs|use|attributesLhs|typeStart|moduleStart|programStart|functionStart|subroutineStart)
    return index

def __resolveDependencies_body(i,index):
    def ascend(module):
        nonlocal index
        nonlocal i
        for used in module["usedModulesOrParentSubprograms"]:
            name = used["name"]
            only = used["only"]
            usedModule = next((m for m in index if m["name"] == name),None)
            if usedModule != None:
                ascend(usedModule)
                if len(only):
                    variables  = []
                    types      = []
                    for var in usedModule["variables"]:
                        if var["name"] in only:
                            var["name"] = only[var["name"]]
                            variables.append(var)
                    for struct in usedModule["types"]:
                        if struct["name"] in only:
                            struct["name"] = only[struct["name"]]
                            types.append(struct)
                else:
                    variables = usedModule["variables"]
                    types     = usedModule["types"]
                localVarNames  = [var["name"] for var in index[i]["variables"]]
                localTypeNames = [typ["name"] for typ in index[i]["types"]]
                index[i]["variables"] = [var for var in variables if var["name"] not in localVarNames]  +\
                   index[i]["variables"]
                index[i]["types"]     = [typ for typ in types     if typ["type"] not in localTypeNames] +\
                   index[i]["types"]  
    ascend(index[i])
    # resolve dependencies
    return i,index[i]

# API
def scanSearchDirs(searchDirs,optionsAsStr):
    index = []
    inputFiles = []
    for searchDir in searchDirs:
        if os.path.exists(searchDir):
            inputFiles += __discoverInputFiles(searchDir)
        else:
            msg = "indexer: include directory '{}' does not exist. Is ignored.".format(searchDir)
            logging.getLogger("").warn(msg); print("WARNING: "+msg,file=sys.stderr)
    partialResults = []
    with Pool(processes=len(inputFiles)) as pool: #untuned
        fileLines = [__readFortranFile(inputFile,optionsAsStr) for i,inputFile in enumerate(inputFiles)]
        partialResults = [pool.apply_async(__parseFile, (fileLines[i],inputFile,)) for i,inputFile in enumerate(inputFiles)]
        pool.close()
        pool.join()
    for p in partialResults:
        index += p.get()
    return index

def dependencyGraphs(index):
    # discover root nodes (serial)
    discoveredModuleNames = [module["name"] for module in index]
    graphs = []
    for module in index:
        isNotRoot = False
        for usedModule in module["usedModulesOrParentSubprograms"]:
            isNotRoot = isNotRoot or usedModule["name"] in discoveredModuleNames
        if not isNotRoot:
            graphs.append(__Node(module["name"],data=module))
    # build tree (parallel)
    handles = []
    with Pool(processes=max(1,int(len(graphs)/2))) as pool: # untuned
         handles = [pool.apply_async(__dependencyGraphs_descend, (root,index,)) for root in graphs]
         pool.close()
         pool.join()
         for i,h in enumerate(handles):
            graphs[i] = h.get()
    return graphs

def resolveDependencies(index,searchedFiles=[],searchedTags=[]):
    def select(module):
        nonlocal searchedFiles
        nonlocal searchedTags
        considerFile = not len(searchedFiles) or module["file"] in searchedFiles
        considerTag  = not len(searchedTags) or module["tag"] in searchedTags
        return considerFile and considerTag

    selection = [i for i,module in enumerate(index) if select(module) and len(module["usedModulesOrParentSubprograms"])]
 
    if len(selection):
        with Pool(processes=len(selection)) as pool: # untuned
            handles = [pool.apply_async(__resolveDependencies_body, (i,index,)) for i in selection]
            pool.close()
            pool.join()
            for h in handles:
                i,result = h.get()
                index[i] = result
    # filter out not needed entries 
    return [module for module in index if select(module)]
