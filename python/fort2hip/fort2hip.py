# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
import os
import utils
import copy
import logging

import addtoplevelpath
import fort2hip.model as model
import translator.translator as translator
import indexer.indexer as indexer
import indexer.indexertools as indexertools
import scanner.scanner as scanner

fort2hipDir = os.path.dirname(__file__)
exec(open("{0}/fort2hip_options.py.in".format(fort2hipDir)).read())

# must be synced with grammar
CLAUSE_NOT_FOUND           = -2
CLAUSE_VALUE_NOT_SPECIFIED = -1

def convertDim3(dim3,dimensions,doFilter=True):
     result = []
     specified = dim3
     if doFilter:
         specified = [ x for x in dim3 if x not in [CLAUSE_NOT_FOUND,CLAUSE_VALUE_NOT_SPECIFIED]]
     for i,value in enumerate(specified):
          if i >= dimensions:
              break
          el = {}
          el["dim"]   = chr(ord("X")+i)
          el["value"] = value
          result.append(el)
     return result

# arg for kernel generator
# array is split into multiple args
def initArg(argName,fType,kind,qualifiers=[],cType="",isArray=False):
    fTypeFinal = fType
    if len(kind):
        fTypeFinal += "({})".format(kind)
    arg = {
      "name"            : argName.replace("%","_") ,
      "callArgName"     : argName,
      "qualifiers"      : qualifiers,
      "type"            : fTypeFinal,
      "origType"        : fTypeFinal,
      "cType"           : cType,
      "cSize"           : "",
      "cValue"          : "",
      "cSuffix"         : "[]" if isArray else "",
      "reductionOp"     : "",
      "bytesPerElement" : translator.bytes(fType,kind,default="-1")
    }
    if not len(cType):
        arg["cType"] = translator.convertToCType(fType,kind,"void")
    return arg

def createArgumentContext(indexedVariable,argName,isLoopKernelArg=False,cudaFortran=False):
    """
    Create an argument context dictionary based on a indexed variable.

    :param indexedVariable: A variable description provided by the indexer.
    :type indexedVariable: STDeclaration
    :return: a dicts containing Fortran `type` and `qualifiers` (`type`, `qualifiers`), C type (`cType`), and `name` of the argument
    :rtype: dict
    """
    arg = initArg(argName,indexedVariable["fType"],indexedVariable["kind"],[ "value" ],"",indexedVariable["rank"]>0)
    if indexedVariable["parameter"] and not indexedVariable["value"] is None:
        arg["cValue"] = indexedVariable["value"] 
    lowerBoundArgs = []  # additional arguments that we introduce if variable is an array
    countArgs      = []
    macro          = None
    # treat arrays
    rank = indexedVariable["rank"] 
    if rank > 0:
        if cudaFortran:
            arg["callArgName"] = "c_loc({})".format(argName) # TODO
        else: # acc
            arg["callArgName"] = scanner.devVarName(argName) # TODO
        arg["type"]       = "type(c_ptr)"
        arg["qualifiers"] = [ "value" ]
        for d in range(1,rank+1):
             # lower bounds
             boundArg = initArg("{}_lb{}".format(argName,d),"integer","c_int",["value","intent(in)"],"const int")
             boundArg["callArgName"] = "lbound({},{})".format(argName,d)
             lowerBoundArgs.append(boundArg)
             # number of elements per dimensions
             countArg = initArg("{}_n{}".format(argName,d),"integer","c_int",["value","intent(in)"],"const int")
             countArg["callArgName"] = "size({},{})".format(argName,d)
             countArgs.append(countArg)
        # create macro expression
        if isLoopKernelArg and not indexedVariable["unspecifiedBounds"]:
            macro = { "expr" : indexedVariable["indexMacro"] }
        else:
            macro = { "expr" : indexedVariable["indexMacroWithPlaceHolders"] }
    return arg, lowerBoundArgs, countArgs, macro

def deriveKernelArguments(index, identifiers, localVars, loopVars, whiteList=[], isLoopKernelArg=False, cudaFortran=False):
    """
    #TODO how to handle struct members?
    """
    kernelArgs          = []
    unknownArgs         = []
    cKernelLocalVars    = []
    macros              = []
    localArgs           = []
    localCpuRoutineArgs = []
    inputArrays         = []

    def includeArgument(name):
        nameLower = name.lower().strip()
        if len(whiteList):
            return name in whiteList
        else:
            if nameLower.startswith("_"):
                return False
            else:
                return True
            # TODO hack. This should be filtered differently. These are local loop variables

    #print(identifiers) # TODO check; should be all lower case
    for name in identifiers: # TODO does not play well with structs
        if includeArgument(name):
            foundDeclaration = name in loopVars # TODO rename loop variables to local variables; this way we can filter out local subroutine variables
            indexedVariable,discovered = indexertools.searchIndexForVariable(index,name) # TODO treat implicit here
            argName = name
            if not discovered:
                 arg = initArg(name,"TODO declaration not found","",[],"TODO declaration not found")
                 unknownArgs.append(arg)
            else:
                arg, lowerBoundArgs, countArgs, macro = createArgumentContext(indexedVariable,name,cudaFortran)
                argName = name.lower().replace("%","_") # TODO
                # modify argument
                if argName in loopVars: # specific for loop kernels
                    arg["qualifiers"]=[]
                    localCpuRoutineArgs.append(arg)
                elif argName in localVars:
                    arg["qualifiers"]=[]
                    if indexedVariable["rank"] > 0:
                        arg["cSize"] = indexedVariable["totalCount"]
                    localCpuRoutineArgs.append(arg)
                    cKernelLocalVars.append(arg)
                else:
                    rank = indexedVariable["rank"]
                    if rank > 0: # specific for cufLoopKernel
                        inputArrays.append({ "name" : name, "rank" : rank })
                        arg["cSize"]    = ""
                        dimensions = "dimension({0})".format(",".join([":"]*rank))
                        # Fortran size expression for allocate
                        fSize = []
                        for i in range(0,rank):
                            fSize.append("{lb}:{lb}+{siz}-1".format(\
                                lb=lowerBoundArgs[i]["name"],siz=countArgs[i]["name"]))
                        localCpuRoutineArgs.append(\
                          { "name" : name,
                            "type" : arg["origType"],
                            "qualifiers" : ["allocatable",dimensions,"target"],
                            "bounds" : ",".join(fSize),
                            "bytesPerElement" : arg["bytesPerElement"]
                          }\
                        )
                    kernelArgs.append(arg)
                    for countArg in countArgs:
                        kernelArgs.append(countArg)
                    for boundArg in lowerBoundArgs:
                        kernelArgs.append(boundArg)
                if not macro is None:
                    macros.append(macro)

    # remove unknown arguments that are actually bound variables
    for unkernelArg in unknownArgs:
        append = True
        for kernelArg in kernelArgs:
            if unkernelArg["name"].lower() == kernelArg["name"].lower():
                append = False
                break
        if append:
            kernelArgs.append(unkernelArg)

    return kernelArgs, cKernelLocalVars, macros, inputArrays, localCpuRoutineArgs
    
def updateContextFromLoopKernels(loopKernels,index,hipContext,fContext):
    """
    loopKernels is a list of STCufLoopKernel objects.
    hipContext, fContext are inout arguments for generating C/Fortran files, respectively.
    """
    hipContext["haveReductions"] = False
    for stkernel in loopKernels:
        parentTag     = stkernel._parent.tag()
        filteredIndex = indexertools.filterIndexByTag(index,parentTag)
       
        fSnippet = "".join(stkernel.lines())

        # translate and analyze kernels
        kernelParseResult = translator.parseLoopKernel(fSnippet,filteredIndex)

        kernelArgs, cKernelLocalVars, macros, inputArrays, localCpuRoutineArgs =\
          deriveKernelArguments(index,\
            kernelParseResult.identifiersInBody(),\
            kernelParseResult.localScalars(),\
            kernelParseResult.loopVars(),\
            [], True, type(stkernel) is scanner.STCufLoopKernel)

        # general
        kernelName         = stkernel.kernelName()
        kernelLauncherName = stkernel.kernelLauncherName()
   
        # treat reductionVars vars / acc default(present) vars
        hipContext["haveReductions"] = False # |= len(reductionOps)
        kernelCallArgNames = []
        reductions = kernelParseResult.gangTeamReductions(translator.makeCStr)
        reductionVars = []
        for arg in kernelArgs:
            name  = arg["name"]
            cType = arg["cType"]
            isReductionVar = False
            for op,variables in reductions.items():
                if name.lower() in [var.lower() for var in variables]:
                    # modify argument
                    arg["qualifiers"].remove("value")
                    arg["cType"] = cType + "*"
                    # reductionVars buffer var
                    bufferName = "_d_" + name
                    var = { "buffer": bufferName, "name" : name, "type" : cType, "op" : op }
                    reductionVars.append(var)
                    # call args
                    kernelCallArgNames.append(bufferName)
                    isReductionVar = True
            if not isReductionVar:
                kernelCallArgNames.append(name)
                if type(stkernel) is scanner.STAccLoopKernel:
                    if len(arg["cSize"]):
                        stkernel.appendDefaultPresentVar(name)
            hipContext["haveReductions"] |= isReductionVar
        # C LoopKernel
        dimensions  = kernelParseResult.numDimensions()
        block = convertDim3(kernelParseResult.numThreadsInBlock(),dimensions)
        if not len(block):
            defaultBlockSize = DEFAULT_BLOCK_SIZES 
            block = convertDim3(defaultBlockSize[dimensions],dimensions)
        hipKernelDict = {}
        hipKernelDict["isLoopKernel"]       = True
        hipKernelDict["size"]               = convertDim3(kernelParseResult.problemSize(),dimensions,doFilter=False)
        hipKernelDict["grid"]               = convertDim3(kernelParseResult.numGangsTeamsBlocks(),dimensions)
        hipKernelDict["block"]              = block
        hipKernelDict["launchBounds"]       = "__launch_bounds__({})".format(DEFAULT_LAUNCH_BOUNDS)
        hipKernelDict["gridDims"  ]         = [ "{}_grid{}".format(kernelName,x["dim"])  for x in block ] # grid might not be always defined
        hipKernelDict["blockDims"  ]        = [ "{}_block{}".format(kernelName,x["dim"]) for x in block ]
        hipKernelDict["kernelName"]         = kernelName
        hipKernelDict["macros"]             = macros
        hipKernelDict["cBody"]              = kernelParseResult.cStr()
        hipKernelDict["fBody"]              = utils.prettifyFCode(fSnippet)
        hipKernelDict["kernelArgs"]         = ["{} {}{}{}".format(a["cType"],a["name"],a["cSize"],a["cSuffix"]) for a in kernelArgs]
        hipKernelDict["kernelCallArgNames"] = kernelCallArgNames
        hipKernelDict["reductions"]         = reductionVars
        hipKernelDict["kernelLocalVars"]    = ["{} {}{}".format(a["cType"],a["name"],a["cSize"]) for a in cKernelLocalVars]
        hipKernelDict["interfaceName"]      = kernelLauncherName
        hipKernelDict["interfaceComment"]   = "" # kernelLaunchInfo.cStr()
        hipKernelDict["interfaceArgs"]      = hipKernelDict["kernelArgs"]
        hipKernelDict["interfaceArgNames"]  = [arg["name"] for arg in kernelArgs] # excludes the stream;
        hipKernelDict["inputArrays"]        = inputArrays
        #inoutArraysInBody                   = [name.lower for name in kernelParseResult.inoutArraysInBody()]
        #hipKernelDict["outputArrays"]       = [array for array in inputArrays if array.lower() in inoutArraysInBody]
        hipKernelDict["outputArrays"]       = inputArrays
        hipContext["kernels"].append(hipKernelDict)

        # Fortran interface with automatic derivation of stkernel launch parameters
        fInterfaceDictAuto = {}
        fInterfaceDictAuto["cName"]    = kernelLauncherName + "_auto"
        fInterfaceDictAuto["fName"]    = kernelLauncherName + "_auto"
        fInterfaceDictAuto["type"]     = "subroutine"
        fInterfaceDictAuto["args"]     = [
          {"type" : "integer(c_int)", "qualifiers" : ["value", "intent(in)"], "name" : "sharedMem", "cSize" : "" },
          {"type" : "type(c_ptr)"   , "qualifiers" : ["value", "intent(in)"], "name" : "stream",   "cSize": ""},
        ]
        fInterfaceDictAuto["args"]    += kernelArgs
        fInterfaceDictAuto["argNames"] = [arg["name"] for arg in fInterfaceDictAuto["args"]]

        # for test
        fInterfaceDictAuto["doTest"]   = False # True
        fInterfaceDictAuto["testComment"] = ["Fortran implementation:"] + fSnippet.split("\n")
        #fInterfaceDictAuto["testComment"] = ["","Hints:","Device variables in scope:"] + ["".join(declared._lines).lower() for declared in deviceVarsInScope]

        #######################################################################
        # Feed argument names back to STLoopKernel for host code modification
        #######################################################################
        stkernel._kernelArgNames = [arg["callArgName"] for arg in kernelArgs]
        stkernel._stream         = kernelParseResult.stream()
        stkernel._sharedMem      = kernelParseResult.sharedMem()

        # Fortran interface with manual specification of stkernel launch parameters
        fInterfaceDictManual = copy.deepcopy(fInterfaceDictAuto)
        fInterfaceDictManual["cName"] = kernelLauncherName
        fInterfaceDictManual["fName"] = kernelLauncherName
        fInterfaceDictManual["args"] = [
          {"type" : "type(dim3)", "qualifiers" : ["intent(in)"], "name" : "grid", "cSize": ""},
          {"type" : "type(dim3)", "qualifiers" : ["intent(in)"], "name" : "block", "cSize": ""},
          {"type" : "integer(c_int)", "qualifiers" : ["intent(in)"],         "name" : "sharedMem", "cSize" : "" },
          {"type" : "type(c_ptr)"   , "qualifiers" : ["value", "intent(in)"], "name" : "stream",   "cSize": ""},
        ]
        fInterfaceDictManual["args"]    += kernelArgs
        fInterfaceDictManual["argNames"] = [arg["name"] for arg in fInterfaceDictManual["args"]]
        fInterfaceDictManual["doTest"]   = False

        # CPU routine
        fRoutineDict = copy.deepcopy(fInterfaceDictAuto)
        fRoutineDict["fName"] = kernelLauncherName + "_cpu"
        
        # rename copied modified args
        for i,val in enumerate(fRoutineDict["args"]):
            varName = val["name"]
            if val.get("cSuffix","")=="[]": # is array
                fRoutineDict["args"][i]["name"] = "d_{}".format(varName)

        fRoutineDict["argNames"] = [a["name"] for a in fRoutineDict["args"]]
        fRoutineDict["args"]    += localCpuRoutineArgs # ordering important
        # add mallocs, memcpys , frees
        prolog = ""
        epilog = ""
        for arg in localCpuRoutineArgs:
             if len(arg.get("bounds","")): # is local Fortran array
               localArray = arg["name"]
               # device to host
               prolog += "allocate({var}({bounds}))\n".format(var=localArray,bounds=arg["bounds"])
               prolog += "CALL hipCheck(hipMemcpy(c_loc({var}),d_{var},{bpe}_8*SIZE({var}),hipMemcpyDeviceToHost))\n".format(var=localArray,bpe=arg["bytesPerElement"])
               # host to device
               epilog += "CALL hipCheck(hipMemcpy(d_{var},c_loc({var}),{bpe}_8*SIZE({var}),hipMemcpyHostToDevice))\n".format(var=localArray,bpe=arg["bytesPerElement"])
               epilog += "deallocate({var})\n".format(var=localArray)
        fRoutineDict["body"] = prolog + fSnippet + epilog

        # Add all definitions to context
        fContext["interfaces"].append(fInterfaceDictManual)
        fContext["interfaces"].append(fInterfaceDictAuto)
        fContext["routines"].append(fRoutineDict)

# TODO check if this can be combined with other routine
def updateContextFromAcceleratorRoutines(acceleratorRoutines,hipContext,fContext):
    """
    acceleratorRoutines is a list of STSubroutine objects.
    hipContext, fContext are inout arguments for generating C/Fortran files, respectively.
    """
    for stroutine in acceleratorRoutines:
        fSnippet = "".join(stroutine._lines)
        fSnippet = utils.prettifyFCode(fSnippet)

        kernelName, argNames, cBody = translator.convertAcceleratorRoutine(fSnippet)
        kernelLauncherName = "launch_{}".format(kernelName)
        loopVars = []; localLValues = []
        
        filteredIndex = indexertools.filterIndexByTag(index,stroutine.tag())

        identifiers = [] # TODO identifiers not the best name # TODO this is redundant with the ignore list
        for declaration in declaredVars:
             identifiers += declaration._vars

        kernelArgs, cKernelLocalVars, macros, localCpuRoutineArgs, localCpuRoutineArrayNames =\
          deriveKernelArguments(filteredIndex,identifiers,localLValues,loopVars,argNames,False,cudaFortran=True)
        #print(argNames)

        def beginOfBody(lines):
            lineno = 0
            while(not "use" in lines[lineno].lower() and\
                  not "implicit" in lines[lineno].lower() and\
                  not "::" in lines[lineno].lower()):
                lineno += 1
            return lineno
        def endOfBody(lines):
            lineno = len(lines)-1
            while(not "end" in lines[lineno].lower()):
                lineno -= 1
            return lineno
        fBody = "".join(stroutine._lines[beginOfBody(stroutine._lines):endOfBody(stroutine._lines)])
        fBody = utils.prettifyFCode(fBody)

        # C  routine and C stroutine launcher
        hipKernelDict = {}
        hipKernelDict["isLoopKernel"]     = False
        hipKernelDict["kernelName"]       = kernelName
        hipKernelDict["macros"]           = macros
        hipKernelDict["cBody"]            = cBody
        hipKernelDict["fBody"]            = fBody
        hipKernelDict["kernelArgs"]       = ["{} {}".format(a["cType"],a["name"]) for a in kernelArgs]
        hipKernelDict["kernelLocalVars"]  = ["{0} {1}{2} {3}".format(a["cType"],a["name"],a["cSize"],"= " + a["cValue"] if "cValue" in a else "") for a in cKernelLocalVars]
        hipKernelDict["interfaceName"]    = kernelLauncherName
        hipKernelDict["interfaceArgs"]    = hipKernelDict["kernelArgs"]
        hipKernelDict["interfaceComment"] = ""
        hipKernelDict["interfaceArgNames"] = [arg["name"] for arg in kernelArgs]
        hipKernelDict["kernelArgs"] = ["{} {}".format(a["cType"],a["name"]) for a in kernelArgs]
        # Fortran interface with manual specification of kernel launch parameters
        fInterfaceDictManual = {}
        fInterfaceDictManual["cName"]       = kernelLauncherName
        fInterfaceDictManual["fName"]       = kernelLauncherName
        fInterfaceDictManual["testComment"] = ["Fortran implementation:"] + fSnippet.split("\n")
        fInterfaceDictManual["type"]        = "subroutine"
        fInterfaceDictManual["args"] = [
            {"type" : "type(dim3)", "qualifiers" : ["intent(in)"], "name" : "grid"},
            {"type" : "type(dim3)", "qualifiers" : ["intent(in)"], "name" : "block"},
            {"type" : "integer(c_int)", "qualifiers" : ["value", "intent(in)"], "name" : "sharedMem"},
            {"type" : "type(c_ptr)", "qualifiers" : ["value", "intent(in)"], "name" : "stream"},
        ]
        fInterfaceDictManual["args"]    += kernelArgs
        fInterfaceDictManual["argNames"] = [arg["name"] for arg in fInterfaceDictManual["args"]]
        fInterfaceDictManual["doTest"]   = True

        # CPU routine
        fRoutineDict = copy.deepcopy(fInterfaceDictManual)
        fRoutineDict["fName"]    = kernelLauncherName + "_cpu" # no cName here
        fRoutineDict["args"]     = kernelArgs
        fRoutineDict["argNames"] = argNames(fRoutineDict["args"])
        # rename copied modified args
        #print(localCpuRoutineArrayNames)
        for i,val in enumerate(fRoutineDict["args"]):
            varName = val["name"]
            #print(varName)
            if varName in localCpuRoutineArrayNames:
                fRoutineDict["args"][i]["name"] = "_{}".format(varName)
        fRoutineDict["argNames"] = ["{}".format(a["name"]) for a in fRoutineDict["args"]]
        fRoutineDict["args"]  += localCpuRoutineArgs # ordering important
        # add memcpys
        prolog = ""
        epilog = ""
        for localCpuRoutineArray in localCpuRoutineArrayNames:
             # hipMemcpy(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind)
             # device to host
             prolog += "CALL hipCheck(hipMemcpy(c_loc({0}),_{0},C_SIZEOF({0}),hipMemcpyDeviceToHost))\n".format(localCpuRoutineArray)
             # host to device
             epilog += "CALL hipCheck(hipMemcpy(_{0},c_loc({0}),C_SIZEOF({0}),hipMemcpyHostToDevice))\n".format(localCpuRoutineArray)
        fRoutineDict["body"] = prolog + fBody + epilog

        # Add all definitions to context
        hipContext["kernels"].append(hipKernelDict)
        fContext["interfaces"].append(fInterfaceDictManual)
        fContext["routines"].append(fRoutineDict)

def renderTemplates(outputFilePrefix,hipContext,fContext):
    # HIP kernel file
    #pprint.pprint(hipContext)
    hipImplementationFilePath = "{0}.kernels.hip.cpp".format(outputFilePrefix)
    model.HipImplementationModel().generateCode(hipImplementationFilePath,hipContext)
    utils.prettifyCFile(hipImplementationFilePath)
    msg = "created HIP C++ implementation file: ".ljust(40) + hipImplementationFilePath
    logger = logging.getLogger("")
    logger.info(msg) ; print(msg)

    # header files
    outputDir = os.path.dirname(hipImplementationFilePath)
    gpufortHeaderFilePath = outputDir + "/gpufort.h"
    model.GpufortHeaderModel().generateCode(gpufortHeaderFilePath)
    msg = "created gpufort main header: ".ljust(40) + gpufortHeaderFilePath
    logger = logging.getLogger("")
    logger.info(msg) ; print(msg)
    if hipContext["haveReductions"]:
        gpufortReductionsHeaderFilePath = outputDir + "/gpufort_reductions.h"
        model.GpufortReductionsHeaderModel().generateCode(gpufortReductionsHeaderFilePath)
        msg = "created gpufort reductions header file: ".ljust(40) + gpufortReductionsHeaderFilePath
        logger = logging.getLogger("")
        logger.info(msg) ; print(msg)

    # Fortran interface/testing module
    moduleFilePath = "{0}.kernels.f08".format(outputFilePrefix)
    model.InterfaceModuleModel().generateCode(moduleFilePath,fContext)
    #utils.prettifyFFile(moduleFilePath)
    msg = "created interface/testing module: ".ljust(40) + moduleFilePath
    logger.info(msg) ; print(msg)

    # TODO disable tests for now
    if False:
       # Fortran test program
       testFilePath = "{0}.kernels.TEST.f08".format(outputFilePrefix)
       model.InterfaceModuleTestModel().generateCode(testFilePath,fContext)
       #utils.prettifyFFile(testFilePath)
       msg = "created interface module test file: ".ljust(40) + testFilePath
       logger.info(msg)
       print(msg)

def createHipKernels(stree,index,kernelsToConvertToHip,outputFilePrefix,basename,generateCode):
    """
    :param stree:        [inout] the scanner tree holds nodes that store the Fortran code lines of the kernels
    :param generateCode: generate code or just feed kernel signature information
                         back to the scanner tree.
    :note The signatures of the identified kernels must be fed back to the 
          scanner tree even when no kernel files are written.
    """
    global FORTRAN_MODULE_PREAMBLE
    if not len(kernelsToConvertToHip):
        return
    
    def select(kernel):
        nonlocal kernelsToConvertToHip
        condition1 = not kernel._ignoreInS2STranslation
        condition2 = \
                kernelsToConvertToHip[0] == "*" or\
                kernel._lineno in kernelsToConvertToHip or\
                kernel.kernelName() in kernelsToConvertToHip
        return condition1 and condition2

    # Context for HIP implementation
    hipContext = {}
    hipContext["includes"] = [ "hip/hip_runtime.h", "hip/hip_complex.h" ]
    hipContext["kernels"] = []
    
    # Context for Fortran interface/implementation
    fContext = {}
    moduleName = basename.replace(".","_").replace("-","_") + "_kernels"
    fContext["name"] = moduleName
    fContext["preamble"] = FORTRAN_MODULE_PREAMBLE
    fContext["used"] = ["hipfort","hipfort_check"]
    fContext["interfaces"] = []
    fContext["routines"]   = []

    # extract kernels
    loopKernels         = stree.findAll(filter=lambda child : isinstance(child, scanner.STLoopKernel) and select(child), recursively=True)
    acceleratorRoutines = stree.findAll(filter=lambda child : type(child) is scanner.STSubroutine and child.isAcceleratorRoutine() and select(child), recursively=True)

    if (len(loopKernels) or len(acceleratorRoutines)):
        updateContextFromLoopKernels(loopKernels,index,hipContext,fContext)
        #updateContextFromAcceleratorRoutines(acceleratorRoutines,hipContext,fContext)
        if generateCode:
            renderTemplates(outputFilePrefix,hipContext,fContext)
