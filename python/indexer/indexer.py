# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
#!/usr/bin/env python3
import addtoplevelpath
import os,sys,subprocess
import re
import threading
import concurrent.futures

import orjson

import translator.translator as translator
import utils.logging

GPUFORT_MODULE_FILE_SUFFIX=".gpufort_mod"

CASELESS    = False
GRAMMAR_DIR = os.path.join(os.path.dirname(__file__),"../grammar")
exec(open("{0}/grammar.py".format(GRAMMAR_DIR)).read())

# configurable parameters
indexerDir = os.path.dirname(__file__)
exec(open("{0}/indexer_options.py.in".format(indexerDir)).read())
    
pFilter       = re.compile(FILTER) 
pAntiFilter   = re.compile(ANTIFILTER)
pContinuation = re.compile(CONTINUATION_FILTER)

def __readFortranFile(filepath,preprocOptions):
    """
    Read and preprocess a Fortran file. Make all
    statements take a single line, i.e. remove all occurences
    of "&".
    """
    global PREPROCESS_FORTRAN_FILE
    global pFilter
    global pAntiFilter
    global pContinuation
    global LOG_PREFIX

    utils.logging.logEnterFunction(LOG_PREFIX,"__readFortranFile",{"filepath":filepath,"preprocOptions":preprocOptions})
    
    def considerLine(strippedLine):
        passesFilter     = pFilter.match(strippedLine) != None
        passesAntifilter = pAntiFilter.match(strippedLine) != None
        utils.logging.logDebug4(LOG_PREFIX,"__readFortranFile","statement '{}' passes select filter: '{}'; statement passes ignore filter: '{}'".format(strippedLine,passesFilter,passesAntifilter))
        return passesFilter and not passesAntifilter
    try:
       command = PREPROCESS_FORTRAN_FILE.format(file=filepath,options=preprocOptions)
       output  = subprocess.check_output(command,shell=True).decode("UTF-8")
       # remove Fortran line continuation and directive continuation
       output = pContinuation.sub(" ",output.lower()) 
       
       # filter statements
       filteredLines = []
       for line in output.split("\n"):
           strippedLine = line.strip().rstrip("\n")
           if considerLine(strippedLine):
               utils.logging.logDebug3(LOG_PREFIX,"__readFortranFile","select statement '{}'".format(strippedLine))
               filteredLines.append(strippedLine)
           else:
               utils.logging.logDebug3(LOG_PREFIX,"__readFortranFile","ignore statement '{}'".format(strippedLine))
    except subprocess.CalledProcessError as cpe:
        raise cpe
    
    utils.logging.logLeaveFunction(LOG_PREFIX,"__readFortranFile")
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
    global PARSE_VARIABLE_DECLARATIONS_WORKER_POOL_SIZE
    global PARSE_VARIABLE_MODIFICATION_STATEMENTS_WORKER_POOL_SIZE 

    utils.logging.logEnterFunction(LOG_PREFIX,"__parseFile",{"filepath":filepath})
    # Regex
    datatype_reg = Regex(r"\b(type\s*\(|character|integer|logical|real|complex|double\s+precision)\b")

    index = []

    # Currently, we are only interested in a modules declarations
    # and its derived types.
    # Later, we might also parse routines

    accessLock   = threading.Lock()
    utils.logging.logDebug(LOG_PREFIX,"__parseFile","create thread pool of size {} for process variable declarations".format(\
      PARSE_VARIABLE_DECLARATIONS_WORKER_POOL_SIZE))
    taskExecutor = concurrent.futures.ThreadPoolExecutor(\
      max_workers=PARSE_VARIABLE_DECLARATIONS_WORKER_POOL_SIZE)
    # statistics
    totalNumTasks = 0
 
    def logEnterJobOrTask_(parentNode,msg):
        utils.logging.logDebug3(LOG_PREFIX,"__parseFile","[thread-id={3}][parent-node={0}:{1}] {2}".format(\
              parentNode._kind, parentNode._name, msg,\
              threading.get_ident()))
        
    def logLeaveJobOrTask_(parentNode,msg):
        utils.logging.logDebug2(LOG_PREFIX+"__parseFile","[thread-id={3}][parent-node={0}:{1}] {2}".format(\
              parentNode._kind, parentNode._name, msg,\
              threading.get_ident()))
    
    def ParseDeclarationTask_(parentNode,inputText):
        """
        :note: the term 'task' should highlight that this is a function
        that is directly submitted to a thread in the worker pool.
        """
        nonlocal accessLock
        msg = "begin to parse variable declaration '{}'".format(inputText)
        logEnterJobOrTask_(parentNode, msg)
        #
        variables =\
          translator.createIndexRecordsFromDeclaration(\
            translator.declaration.parseString(inputText)[0])
        accessLock.acquire()
        parentNode._data["variables"] += variables
        accessLock.release()
        #
        msg = "parsed variable declaration '{}'".format(inputText)
        logLeaveJobOrTask_(parentNode, msg)
    
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
            msg = "begin to parse attributes statement '{}'".format(self._inputText)
            logEnterJobOrTask_(self._parentNode, msg)
            #
            attribute, modifiedVars = \
                translator.parseAttributes(translator.attributes.parseString(self._inputText)[0])
            for varContext in self._parentNode._data["variables"]:
                if varContext["name"] in modifiedVars and attribute in varContext:
                    accessLock.acquire()
                    varContext[attribute] = True
                    accessLock.release()
            #
            msg = "parsed attributes statement '{}'".format(self._inputText)
            logLeaveJobOrTask_(self._parentNode, msg)
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
            msg = "begin to parse acc declare directive '{}'".format(self._inputText)
            logEnterJobOrTask_(self._parentNode, msg)
            #
            parseResult = translator.acc_declare.parseString(self._inputText)[0]
            for varContext in self._parentNode._data["variables"]:
                for varName in parseResult.mapAllocVariables():
                    if varContext["name"] == varName:
                        accessLock.acquire()
                        varContext["declareOnTarget"] = "alloc"
                        accessLock.release()
                for varName in parseResult.mapToVariables():
                    if varContext["name"] == varName: 
                        accessLock.acquire()
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
            msg = "parsed acc declare directive '{}'".format(self._inputText)
            logLeaveJobOrTask_(self._parentNode, msg)

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
        utils.logging.logDebug(LOG_PREFIX,"__parseFile","[current-node={0}:{1}] enter {2} '{3}' in statement: '{4}'".format(\
          currentNode._parent._kind,currentNode._parent._name,
          currentNode._kind,currentNode._name,\
          currentLine))
    def logLeaveNode_():
        nonlocal currentNode
        nonlocal currentLine
        utils.logging.logDebug(LOG_PREFIX,"__parseFile","[current-node={0}:{1}] leave {0} '{1}' in statement: '{2}'".format(\
          currentNode._data["kind"],currentNode._data["name"],\
          currentLine))
    def logDetection_(kind):
        nonlocal currentNode
        nonlocal currentLine
        utils.logging.logDebug2(LOG_PREFIX,"__parseFile","[current-node={}:{}] found {} in statement: '{}'".format(\
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
        global LOG_PREFIX
        nonlocal currentLine
        nonlocal currentNode
        logDetection_("start of subroutine")
        if currentNode._kind in ["module","program","subroutine","function"]:
            name = tokens[1]
            subroutine = createBaseEntry_("subroutine",name,filepath)
            subroutine["attributes"]  = [q.lower() for q in tokens[0]]
            subroutine["dummyArgs"]   = list(tokens[2])
            currentNode._data["subprograms"].append(subroutine)
            currentNode = __Node("subroutine",name,data=subroutine,parent=currentNode)
            logEnterNode_()
        else:
            utils.logging.logWarning(LOG_PREFIX,"__parseFile","found subroutine in '{}' but parent is {}; expected program/module/subroutine/function parent.".\
              format(currentLine,currentNode._kind))
    #host|device,name,[args],result
    def FunctionStart(tokens):
        global LOG_PREFIX
        nonlocal currentLine
        nonlocal currentNode
        logDetection_("start of function")
        if currentNode._kind in ["module","program","subroutine","function"]:
            name = tokens[1]
            function = createBaseEntry_("function",name,filepath)
            function["attributes"]  = [q.lower() for q in tokens[0]]
            function["dummyArgs"]   = list(tokens[2])
            function["resultName"]  = name if tokens[3] is None else tokens[3]
            currentNode._data["subprograms"].append(function)
            currentNode = __Node("function",name,data=function,parent=currentNode)
            logEnterNode_()
        else:
            utils.logging.logWarning(LOG_PREFIX,"__parseFile","found function in '{}' but parent is {}; expected program/module/subroutine/function parent.".\
              format(currentLine,currentNode._kind))
    def TypeStart(tokens):
        global LOG_PREFIX
        nonlocal currentLine
        nonlocal currentNode
        logDetection_("start of type")
        if currentNode._kind in ["module","program"]:
            assert len(tokens) == 2
            name = tokens[1]
            derivedType = {}
            derivedType["name"]      = name
            derivedType["kind"]      = "type"
            derivedType["variables"] = []
            derivedType["types"] = []
            currentNode._data["types"].append(derivedType)
            currentNode = __Node("type",name,data=derivedType,parent=currentNode)
            logEnterNode_()
        else:
            utils.logging.logWarning(LOG_PREFIX,"__parseFile","found derived type in '{}' but parent is {}; expected program or module parent.".\
                    format(currentLine,currentNode._kind))
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
    def Attributes(tokens):
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
    def AccDeclare(tokens):
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

    typeEnd.setParseAction(End)
    structureEnd.setParseAction(End)

    datatype_reg.setParseAction(Declaration)
    use.setParseAction(Use)
    attributes.setParseAction(Attributes)
    
    acc_declare.setParseAction(AccDeclare)

    def tryToParseString(expressionName,expression):
        try:
           expression.parseString(currentLine)
           return True
        except ParseBaseException as e: 
           utils.logging.logDebug3(LOG_PREFIX,"__parseFile","did not find expression '{}' in statement '{}'".format(expressionName,currentLine))
           utils.logging.logDebug4(LOG_PREFIX,"__parseFile",str(e))
           return False

    for currentLine in fileLines:
        utils.logging.logDebug3(LOG_PREFIX,"__parseFile","process statement '{}'".format(currentLine))
        # typeStart must be tried before datatype_reg
        tryToParseString("structureEnd|typeEnd|typeStart|declaration|use|attributes|acc_declare|moduleStart|programStart|functionStart|subroutineStart",\
          typeEnd|structureEnd|typeStart|datatype_reg|use|attributes|acc_declare|moduleStart|programStart|functionStart|subroutineStart)
    taskExecutor.shutdown(wait=True) # waits till all tasks have been completed

    # apply attributes and acc variable modifications
    numPostParsingJobs = len(postParsingJobs)
    if numPostParsingJobs > 0:
        utils.logging.logDebug(LOG_PREFIX,"__parseFile","apply variable modifications (submit {} jobs to worker pool of size {})".format(\
          numPostParsingJobs,PARSE_VARIABLE_MODIFICATION_STATEMENTS_WORKER_POOL_SIZE))
        with concurrent.futures.ThreadPoolExecutor(\
            max_workers=PARSE_VARIABLE_MODIFICATION_STATEMENTS_WORKER_POOL_SIZE)\
                as jobExecutor:
            for job in postParsingJobs:
                jobExecutor.submit(job.run)
        utils.logging.logDebug(LOG_PREFIX,"__parseFile","apply variable modifications --- done") 
        postParsingJobs.clear()

    utils.logging.logLeaveFunction(LOG_PREFIX,"__parseFile") 
    return index

def __writeJsonFile(index,filepath):
    global PRETTY_PRINT_INDEX_FILE
    global LOG_PREFIX    
    utils.logging.logEnterFunction(LOG_PREFIX,"__writeJsonFile",{"filepath":filepath}) 
    
    with open(filepath,"wb") as outfile:
         if PRETTY_PRINT_INDEX_FILE:
             outfile.write(orjson.dumps(index,option=orjson.OPT_INDENT_2))
         else:
             outfile.write(orjson.dumps(index))
    
    utils.logging.logLeaveFunction(LOG_PREFIX,"__writeJsonFile") 

def __readJsonFile(filepath):
    global LOG_PREFIX    
    utils.logging.logEnterFunction(LOG_PREFIX,"__readJsonFile",{"filepath":filepath}) 
    
    with open(filepath,"rb") as infile:
         utils.logging.logLeaveFunction(LOG_PREFIX,"__readJsonFile") 
         return orjson.loads(infile.read())

# API
def scanFile(filepath,preprocOptions,index):
    """
    Creates an index from a single file.
    """
    global LOG_PREFIX
    utils.logging.logEnterFunction(LOG_PREFIX,"scanFile",{"filepath":filepath,"preprocOptions":preprocOptions}) 
    
    filteredLines = __readFortranFile(filepath,preprocOptions)
    utils.logging.logDebug2(LOG_PREFIX,"scanFile","extracted the following lines:\n>>>\n{}\n<<<".format(\
        "\n".join(filteredLines)))
    index += __parseFile(filteredLines,filepath)
    
    utils.logging.logLeaveFunction(LOG_PREFIX,"scanFile") 

def writeGpufortModuleFiles(index,outputDir):
    """
    Per module / program found in the index
    write a GPUFORT module file.
    
    :param list index:    [in] Empty or non-empty list.
    :param str outputDir: [in] Output directory.
    """
    global LOG_PREFIX
    utils.logging.logEnterFunction(LOG_PREFIX,"writeGpufortModuleFiles",{"outputDir":outputDir})
    
    for mod in index:
        filepath = outputDir + "/" + mod["name"] + GPUFORT_MODULE_FILE_SUFFIX
        __writeJsonFile(mod,filepath)
    
    utils.logging.logLeaveFunction(LOG_PREFIX,"writeGpufortModuleFiles")

def loadGpufortModuleFiles(inputDirs,index):
    """
    Load gpufort module files and append to the index.

    :param list inputDirs: [in] List of input directories (as strings).
    :param list index:     [inout] Empty or non-empty list. Loaded data structure is appended.
    """
    global LOG_PREFIX
    utils.logging.logEnterFunction(LOG_PREFIX,"loadGpufortModuleFiles",{"inputDirs":",".join(inputDirs)})
    
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
    
    utils.logging.logLeaveFunction(LOG_PREFIX,"loadGpufortModuleFiles")

