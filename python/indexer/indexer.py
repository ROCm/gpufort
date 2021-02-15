#!/usr/bin/env python3
import addtoplevelpath
import os,sys
import re
import subprocess
import tempfile
import logging
import json
import time

from multiprocessing import Pool

import translator.translator as translator
import utils

grammarDir = os.path.join(os.path.dirname(__file__),"../grammar")
exec(open("{0}/grammar_options.py.in".format(grammarDir)).read())
exec(open("{0}/grammar_f03.py.in".format(grammarDir)).read())
exec(open("{0}/grammar_cuf.py.in".format(grammarDir)).read())
exec(open("{0}/grammar_acc.py.in".format(grammarDir)).read())
exec(open("{0}/grammar_epilog.py.in".format(grammarDir)).read())

# configurable parameters
indexerDir = os.path.dirname(__file__)
exec(open("{0}/indexer_options.py.in".format(indexerDir)).read())
    
pContinuation = re.compile(CONTINUATION)
pFilter       = re.compile(FILTER) 
pAntiFilter   = re.compile(ANTIFILTER)

EMPTY = { "types" : [], "variables" : [] } 

EMPTY_VARIABLE = {
  "name"                       : "UNKNOWN",
  "fType"                      : "UNKNOWN",
  "kind"                       : "UNKNOWN",
  "bytesPerElement"            : "UNKNOWN",
  "cType"                      : "UNKNOWN",
  "fInterfaceType"             : "UNKNOWN",
  "fInterfaceQualifiers"       : "UNKNOWN",
  "hasParameter"               : "UNKNOWN",
  "hasPointer"                 : "UNKNOWN",
  "hasDevice"                  : "UNKNOWN",
  "hasPinned"                  : "UNKNOWN",
  "hasManaged"                 : "UNKNOWN",
  "hasAllocatable"             : "UNKNOWN",
  "declaredOnTarget"           : "UNKNOWN",
  "rank"                       : "UNKNOWN",
  "unspecifiedBounds"          : "UNKNOWN",
  "lbounds"                    : "UNKNOWN",
  "counts"                     : "UNKNOWN",
  "totalCount"                 : "UNKNOWN",
  "totalBytes"                 : "UNKNOWN",
  "indexMacro"                 : "UNKNOWN",
  "indexMacroWithPlaceHolders" : "UNKNOWN"
}

def __readFortranFile(filepath,compilerOptions):
    def considerLine(strippedLine):
        return (pFilter.match(strippedLine) is not None) and (pAntiFilter.match(strippedLine) is None)

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

    def End(tokens):
        nonlocal current
        current = current._parent
    def ModuleStart(tokens):
        nonlocal current
        name = tokens[0]
        module = {}
        module["type"]        = "module"
        module["name"]        = name
        module["tag"]         = name
        module["file"]        = filePath
        module["types"]       = []
        module["variables"]   = []
        module["usedModulesOrParentSubprograms"] = []
        root._data.append(module)
        current = __Node("module",data=module,parent=current)
    def ProgramStart(tokens):
        nonlocal current
        name = tokens[0]
        program = {}
        program["type"]        = "program"
        program["name"]        = name
        program["tag"]         = name
        program["file"]        = filePath
        program["types"]       = []
        program["variables"]   = []
        program["usedModulesOrParentSubprograms"] = []
        root._data.append(program)
        current = __Node("program",data=program,parent=current)
    #host|device,name,[args]
    def SubroutineStart(tokens):
        nonlocal current
        subroutine = {}
        subroutine["type"] = "subroutine"
        subroutine["name"] = tokens[1]
        if current != root:
            subroutine["tag"] = current._data["name"] + ":" + tokens[1]
        else:
            subroutine["tag"] = tokens[1]
        subroutine["hasHost"]     = "host" in tokens[0]
        subroutine["hasDevice"]   = "device" in tokens[0]
        subroutine["hasGlobal"]   = "global" in tokens[0]
        subroutine["dummyArgs"]   = list(tokens[2])
        subroutine["file"]        = filePath
        subroutine["types"]       = []
        subroutine["variables"]   = []
        subroutine["usedModulesOrParentSubprograms"] = []
        if current._name not in ["root","module"]:
            subroutine["usedModulesOrParentSubprograms"].append({ "name": current._data["name"], "only": []})
        root._data.append(subroutine)
        current = __Node("subroutine",data=subroutine,parent=current)
    #host|device,name,[args],result
    def FunctionStart(tokens):
        nonlocal current
        function = {}
        function["type"]        = "function"
        function["name"]        = tokens[1]
        if current != root:
            function["tag"] = current._data["name"] + ":" + tokens[1]
        else:
            function["tag"] = tokens[1]
        function["hasHost"]     = "host" in tokens[0]
        function["hasDevice"]   = "device" in tokens[0]
        function["hasGlobal"]   = "global" in tokens[0]
        function["dummyArgs"]   = list(tokens[2])
        function["resultName"]  = tokens[1] if tokens[3] is None else tokens[3]
        function["file"]        = filePath
        function["types"]       = []
        function["variables"]   = []
        function["usedModulesOrParentSubprograms"] = []
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
          translator.createCodegenContextFromDeclaration(\
            translator.declaration.parseString(currentLine)[0])
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
    # TODO openacc pragmas

    def tryToParseString(expressionName,expression):
        try:
           expression.parseString(currentLine)
           logging.getLogger("").debug("modscan:\tFOUND expression '{}' in line: '{}'".format(expressionName,currentLine))
           return True
        except ParseBaseException as e: 
           logging.getLogger("").debug2("modscan:\tdid not find expression '{}' in line '{}'".format(expressionName,currentLine))
           logging.getLogger("").debug3(str(e))
           return False

    for currentLine in fileLines:
        tryToParseString("structureEnd|typeEnd|declaration|use|typeStart|moduleStart|programStart|functionStart|subroutineStart",\
            typeEnd|structureEnd|declarationLhs|use|typeStart|moduleStart|programStart|functionStart|subroutineStart)
    return index

def __resolveDependencies_body(i,index):
    def ascend(module):
        nonlocal index
        nonlocal i
        for used in module["usedModulesOrParentSubprograms"]:
            name = used["name"]
            only = used["only"]
            usedModule = next((m for m in index if m["name"] == name),None)
            if usedModule is not None:
                ascend(usedModule)
                if len(only):
                    variables = []
                    types     = []
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
                index[i]["variables"] = variables + index[i]["variables"]
                index[i]["types"]     = types     + index[i]["types"]  
    ascend(index[i])
    # resolve dependencies
    return i,index[i]

# API
def scanSearchDirs(searchDirs,optionsAsStr):
    index = []
    inputFiles = []
    for searchDir in searchDirs:
         inputFiles += __discoverInputFiles(searchDir)
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

def filterIndexByTag(index,tag):
    """
    Return only the structure(s) (module,program,subroutine,function) with
    a certain tag.
    """
    resultSet = [structure for structure in index if structure["tag"] == tag]
    if len(resultSet) is not 1:
        msg = "'{}' entries found for tag '{}'. Expected to find a single entry.".format(len(resultSet),tag)
        if indexer.ERROR_HANDLING == "strict":
            logging.getLogger("").error(msg)
            print("ERROR: "+msg,file=sys.stderr)
            sys.exit(1001)
        else:
            logging.getLogger("").warn(msg)
            return [ indexer.EMPTY ]
    else:
        msg = "'{}' entries found for tag '{}'".format(len(resultSet),tag)
        logging.getLogger("").debug2(msg)
        return resultSet

def searchIndexForVariable(index,variableExpression):
    """
    Input might be a simple identifier such as 'a' or 'A_d'
    or a more complicated derived-type member expression such
    as 'a%b%c' or 'A%b(i)%c'.

    :param index: list with single structure dict as produced by 'filterIndexByTag'. 

    :see: filterIndexByTag

    :note: Fortran does not support nested declarations of types. If a derived type
    has other derived type members, they must be declared before the definition of a new
    type that uses them.
    """
    result = None
    def lookupFromLeftToRight(structure,expression):
        nonlocal result
        if "%" not in expression:
            result = next((var for var in structure["variables"] if var["name"] == expression),None)  
        else:
            parts     = expression.split("%")
            typeVar   = parts[0].split("(")[0] # strip away array brackets
            remainder = "%".join(parts[1:])
            try:
                matchingTypeVar = next((var for var in structure["variables"] if var["name"] == typeVar),None)
                matchingType    = next((struct for struct in structure["types"] if struct["name"] == matchingTypeVar["kind"]),None)
                lookupFromLeftToRight(matchingType,remainder)
            except:
                pass
    for structure in index:
        lookupFromLeftToRight(structure,variableExpression.lower().replace(" ",""))
    if result is None:
        result         = indexer.EMPTY_VARIABLE
        result["name"] = variableExpression
        msg = "No entry found for variable '{}'.".format(variableExpression)
        if indexer.ERROR_HANDLING == "strict":
            logging.getLogger("").error(msg)
            print("ERROR: "+msg,file=sys.stderr)
            sys.exit(1002)
        else:
            logging.getLogger("").warn(msg)
        return result, False
    else:
        msg = "single entry found for variable '{}'".format(variableExpression)
        logging.getLogger("").debug2(msg)
        return result, True
