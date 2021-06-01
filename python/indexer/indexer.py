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
    
pFilter        = re.compile(FILTER) 
pAntiFilter    = re.compile(ANTIFILTER)
pContinuation  = re.compile(CONTINUATION_FILTER)
pPrescanFilter = re.compile(PRESCAN_FILTER) 

def __readFortranFile(filepath,compilerOptions,prescan=False):
    """
    :param prescan: If this is a pre scan, then only programs,
    """
    def considerLine(strippedLine):
        nonlocal prescan
        if prescan:
            return (pPrescanFilter.match(strippedLine) != None) and (pAntiFilter.match(strippedLine) is None)
        else:
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
        self._data     = data
    def __str__(self):
        return "{}: {}".format(self._name,self._data)
    __repr__ = __str__

def __parseFile(fileLines,filePath,prescan=False):
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
        entry["variables"]                       = []
        entry["subprograms"]                     = []
        entry["types"]                           = []
        entry["usedModules"]  = []
        return entry

    def End(tokens):
        nonlocal root
        nonlocal current
        if current != root:
            current = current._parent
    def ModuleStart(tokens):
        nonlocal root
        nonlocal current
        name = tokens[0]
        module = createBaseEntry_("module",name,name,filePath)
        root._data.append(module)
        current = __Node("module",data=module,parent=current)
    def ProgramStart(tokens):
        nonlocal root
        nonlocal current
        name    = tokens[0]
        program = createBaseEntry_("program",name,name,filePath)
        root._data.append(program)
        current = __Node("program",data=program,parent=current)
    #host|device,name,[args]
    def SubroutineStart(tokens):
        nonlocal root
        nonlocal current
        if current != root:
            name = tokens[1]
            subroutine = createBaseEntry_("subroutine",name,name,filePath)
            if current != root:
                subroutine["tag"] = current._data["name"] + ":" + name
            subroutine["attributes"]  = [q.lower() for q in tokens[0]]
            subroutine["dummyArgs"]   = list(tokens[2])
            current._data["subprograms"].append(subroutine)
            current = __Node("subroutine",data=subroutine,parent=current)
    #host|device,name,[args],result
    def FunctionStart(tokens):
        nonlocal root
        nonlocal current
        if current != root:
            name = tokens[1]
            function = createBaseEntry_("function",name,name,filePath)
            if current != root:
                function["tag"] = current._data["name"] + ":" + name
            function["attributes"]  = [q.lower() for q in tokens[0]]
            function["dummyArgs"]   = list(tokens[2])
            function["resultName"]  = name if tokens[3] is None else tokens[3]
            current._data["subprograms"].append(function)
            current = __Node("function",data=function,parent=current)
    
    def TypeStart(tokens):
        nonlocal root
        nonlocal current
        if current != root:
            name = tokens[1]
            derivedType = {}
            derivedType["name"]      = name
            derivedType["variables"] = []
            current._data["types"].append(derivedType)
            current = __Node("type",data=derivedType,parent=current)
    def Declaration(s,loc,tokens):
        nonlocal root
        nonlocal current
        nonlocal currentLine
        #print(currentLine)
        if current != root:
            current._data["variables"] +=\
              translator.createIndexRecordsFromDeclaration(\
                translator.declaration.parseString(currentLine)[0])
    def Attributes(s,loc,tokens):
        """
        Add attributes to previously declared variables in same scope.
        Does not modify scope of other variables.
        """
        # TODO investigate if target of attribute must be in same scope or not!
        nonlocal root
        nonlocal current
        nonlocal currentLine
        #print(currentLine)
        if current != root:
            attribute, modifiedVars = \
                translator.parseAttributes(translator.attributes.parseString(currentLine)[0])
            for varContext in current._data["variables"]:
                if varContext["name"] in modifiedVars and attribute in varContext:
                    varContext[attribute] = True
    def AccDeclare(s,loc,tokens):
        """
        Add attributes to previously declared variables in same scope.
        Does not modify scope of other variables.
        """
        # TODO investigate if target of attribute must be in same scope or not!
        nonlocal root
        nonlocal current
        nonlocal currentLine
        parseResult = translator.acc_declare.parseString(currentLine)[0]
        if current != root:
            for varContext in current._data["variables"]:
                for varName in parseResult.mapAllocVariables():
                    if varContext["name"] == varName:
                        varContext["declareOnTarget"] = "alloc"
                for varName in parseResult.mapToVariables():
                    if varContext["name"] == varName: 
                        varContext["declareOnTarget"] = "to"
                for varName in parseResult.mapFromVariables():
                    if varContext["name"] == varName: 
                        varContext["declareOnTarget"] = "from"
                for varName in parseResult.mapTofromVariables():
                    if varContext["name"] == varName: 
                        varContext["declareOnTarget"] = "tofrom"
    # 'use kinds, only: dp, sp => sp2' --> [None, 'kinds', [['dp', None], ['sp', 'sp2']]]
    def Use(tokens):
        nonlocal root
        nonlocal current
        if current != root:
            usedModule = {}
            usedModule["name"] = translator.makeFStr(tokens[1])
            usedModule["only"] = {}
            for pair in tokens[2]:
                original = translator.makeFStr(pair[0])
                renaming = original if pair[1] is None else translator.makeFStr(pair[1])
                usedModule["only"][original]=renaming
            #print(current)
            current._data["usedModules"].append(usedModule) # TODO only include what is necessary
    
    moduleStart.setParseAction(ModuleStart)
    typeStart.setParseAction(TypeStart)
    programStart.setParseAction(ProgramStart)
    functionStart.setParseAction(FunctionStart)
    subroutineStart.setParseAction(SubroutineStart)
    acc_declare.setParseAction(AccDeclare)

    typeEnd.setParseAction(End)
    structureEnd.setParseAction(End)

    declarationLhs.setParseAction(Declaration)
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
        if prescan:
            tryToParseString("structureEnd|use|moduleStart|programStart|functionStart|subroutineStart",\
              structureEnd|use|moduleStart|programStart|functionStart|subroutineStart)
        else:
            tryToParseString("structureEnd|typeEnd|declaration|use|attributes|acc_declare|typeStart|moduleStart|programStart|functionStart|subroutineStart",\
              typeEnd|structureEnd|declarationLhs|use|attributesLhs|acc_declare|typeStart|moduleStart|programStart|functionStart|subroutineStart)
    return index

def __resolveDependencies_body(i,index):
    def ascend(module):
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

# API
def scanSearchDirs(searchDirs,optionsAsStr,prescan=False):
    global SCAN_SEARCH_DIRS_MAX_PROCESSES

    index = []
    inputFiles = []
    for searchDir in searchDirs:
        if os.path.exists(searchDir):
            inputFiles += __discoverInputFiles(searchDir)
        else:
            msg = "indexer: include directory '{}' does not exist. Is ignored.".format(searchDir)
            utils.logWarn(msg)
    partialResults = []
    print("\n".join(inputFiles))
    with Pool(processes=min(SCAN_SEARCH_DIRS_MAX_PROCESSES,len(inputFiles))) as pool: #untuned
        fileLines = [__readFortranFile(inputFile,optionsAsStr,prescan) for i,inputFile in enumerate(inputFiles)]
        partialResults = [pool.apply_async(__parseFile, (fileLines[i],inputFile,prescan,)) for i,inputFile in enumerate(inputFiles)]
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
        for usedModule in module["usedModules"]:
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

    selection = [i for i,module in enumerate(index) if select(module) and len(module["usedModules"])]
 
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

def writeIndexToFile(index,filepath):
    with open(filepath,"w") as outfile:
            json.dump(index, indexFile, indent=2)
        #

def loadIndexFromFile(filepath):
    return

