# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
#!/usr/bin/env python3
import addtoplevelpath
import os,sys
import copy

import orjson

import utils.logging

# configurable parameters
indexerDir = os.path.dirname(__file__)
exec(open("{0}/scoper_options.py.in".format(indexerDir)).read())

ERR_SCOPER_RESOLVE_DEPENDENCIES_FAILED = 1001
ERR_SCOPER_LOOKUP_FAILED = 1002

EMPTY_SCOPE = { "tag": "", "types" : [], "variables" : [], "subprograms" : []} 

__UNKNOWN = "UNKNOWN"

EMPTY_VARIABLE = {                         
  "name"                       : __UNKNOWN,
  "fType"                      : __UNKNOWN,
  "kind"                       : __UNKNOWN,
  "bytesPerElement"            : __UNKNOWN,
  "cType"                      : __UNKNOWN,
  "fInterfaceType"             : __UNKNOWN,
  "fInterfaceQualifiers"       : __UNKNOWN,
  "parameter"                  : __UNKNOWN,
  "pointer"                    : __UNKNOWN,
  "device"                     : __UNKNOWN,
  "pinned"                     : __UNKNOWN,
  "managed"                    : __UNKNOWN,
  "allocatable"                : __UNKNOWN,
  "declareOnTarget"            : __UNKNOWN,
  "rank"                       : -1,
  "unspecifiedBounds"          : __UNKNOWN,
  "lbounds"                    : __UNKNOWN,
  "counts"                     : __UNKNOWN,
  "totalCount"                 : __UNKNOWN,
  "totalBytes"                 : __UNKNOWN,
  "indexMacro"                 : __UNKNOWN,
  "indexMacroWithPlaceHolders" : __UNKNOWN
}

EMPTY_TYPE = {                         
  "name"      : __UNKNOWN,
  "variables" : []
}

EMPTY_SUBPROGRAM = {                         
  "kind"        : __UNKNOWN,
  "name"        : __UNKNOWN,
  "resultName"  : __UNKNOWN,
  "variables"   : [],
  "subprograms" : [],
  "usedModules" : []
}

__SCOPE_ENTRY_TYPES = ["subprograms","variables","types"]

def __resolveDependencies(scope,indexRecord,index):
    """
    Include variable, type, and subprogram records from modules used
    by the current record (module,program or subprogram).

    :param dict scope: the scope that you updated with information from the used modules.
    :param dict indexRecord: a module/program/subprogram index record
    :param list index: list of module/program index records

    TODO must be recursive!!!
    """
    global LOG_PREFIX    
    global ERROR_HANDLING

    utils.logging.logEnterFunction(LOG_PREFIX,"__resolveDependencies")

    def handleUseStatements_(scope,moduleRecord):
        """
        recursive function
        :param dict moduleRecord: 
        """ 
        nonlocal index
        for usedModule in moduleRecord["usedModules"]:
            usedModuleFound = usedModule["name"] in MODULE_IGNORE_LIST
            # include definitions from other modules
            for module in index:
                if module["name"] == usedModule["name"]:
                    handleUseStatements_(scope,module) # recursivie call

                    usedModuleFound   = True
                    includeAllEntries = not len(usedModule["only"])
                    if includeAllEntries: # simple include
                        utils.logging.logDebug2(LOG_PREFIX,"__resolveDependencies.handleUseStatements",
                          "use all definitions from module '{}'".format(moduleRecord["name"]))
                        for entryType in __SCOPE_ENTRY_TYPES:
                            scope[entryType] += module[entryType]
                    else:
                        for mapping in usedModule["only"]:
                            for entryType in __SCOPE_ENTRY_TYPES:
                                for entry in module[entryType]:
                                    if entry["name"] == mapping["original"]:
                                        utils.logging.logDebug2(LOG_PREFIX,
                                          "__resolveDependencies.handleUseStatements",\
                                          "use {} '{}' as '{}' from module '{}'".format(\
                                          entryType[0:-1],mapping["original"],mapping["renamed"],\
                                          moduleRecord["name"]))
                                        copiedEntry = copy.deepcopy(entry)
                                        copiedEntry["name"] = mapping["renamed"]
                                        scope[entryType].append(copiedEntry)
            if not usedModuleFound:
                msg = "no index record for module '{}' could be found".format(usedModule["name"])
                if ERROR_HANDLING == "strict":
                    utils.logging.logError(LOG_PREFIX,"__resolveDependencies",msg) 
                    sys.exit(ERR_INDEXER_RESOLVE_DEPENDENCIES_FAILED)
                else:
                    utils.logging.logWarning(LOG_PREFIX,"__resolveDependencies",msg)

    handleUseStatements_(scope,indexRecord)
    utils.logging.logLeaveFunction(LOG_PREFIX,"__resolveDependencies")

def __createScope(index,tag):
    """
    :param str tag: a colon-separated list of strings. Ex: mymod:mysubroutine or mymod.
    :note: not thread-safe
    :note: tries to reuse existing scopes.
    :note: assumes that number of scopes will be small per file. Hence, uses list instead of tree data structure
           for storing scopes.
    """
    global SCOPES
    global REMOVE_OUTDATED_SCOPES
    global MODULE_IGNORE_LIST
    global LOG_PREFIX    
    utils.logging.logEnterFunction(LOG_PREFIX,"__createScope",{"tag":tag,"ERROR_HANDLING":ERROR_HANDLING})
    
    # check if already a scope exists for the tag or if
    # it can be derived from a higher-level scope
    existingScope   = EMPTY_SCOPE
    nestingLevel    = -1 # -1 implies that nothing has been found
    scopesToDelete  = []
    for s in SCOPES:
        existingTag = s["tag"]
        if existingTag == tag[0:len(existingTag)]:
            existingScope = s
            nestingLevel  = len(existingTag.split(":"))-1
        else:
            scopesToDelete.append(s)
    # clean up scopes that are not used anymore 
    if REMOVE_OUTDATED_SCOPES and len(scopesToDelete):
        utils.logging.logDebug(LOG_PREFIX,"__createScope",\
          "delete outdated scopes with tags '{}'".format(\
            ", ".join([s["tag"] for s in scopesToDelete])))
        for s in scopesToDelete:
            SCOPES.remove(s)

    # return existing existingScope or create it
    tagTokens = tag.split(":")
    if len(tagTokens)-1 == nestingLevel:
        utils.logging.logDebug(LOG_PREFIX,"__createScope",\
          "found existing scope for tag '{}'".format(tag))
        utils.logging.logLeaveFunction(LOG_PREFIX,"__createScope")
        return existingScope
    else:
        newScope = copy.deepcopy(existingScope)
        newScope["tag"] = tag 
 
        # we already have a scope for this record
        if nestingLevel >= 0:
            baseRecordTag = ":".join(tagTokens[0:nestingLevel+1])
            utils.logging.logDebug(LOG_PREFIX,"__createScope",\
              "create scope for tag '{}' based on existing scope with tag '{}'".format(tag,baseRecordTag))
            baseRecord = next((module for module in index if module["name"] == tagTokens[0]),None)  
            for l in range(1,nestingLevel+1):
                baseRecord = next((subprogram for subprogram in baseRecord["subprograms"] if subprogram["name"] == tagTokens[l]),None)
            currentRecordList = baseRecord["subprograms"]
        else:
            utils.logging.logDebug(LOG_PREFIX,"__createScope",\
              "create scope for tag '{}'".format(tag))
            currentRecordList = index
        begin = nestingLevel + 1 # 
        
        for d in range(begin,len(tagTokens)):
            searchedName = tagTokens[d]
            for currentRecord in currentRecordList:
                if currentRecord["name"] == searchedName:
                    # 1. first include variables from included
                    __resolveDependencies(newScope,currentRecord,index) 
                    # 2. now include the current record's   
                    for entryType in __SCOPE_ENTRY_TYPES:
                        if entryType in currentRecord:
                            newScope[entryType] += currentRecord[entryType]
                    currentRecordList = currentRecord["subprograms"]
                    break
        SCOPES.append(newScope)
        utils.logging.logLeaveFunction(LOG_PREFIX,"__createScope")
        return newScope

def __searchIndexForTypeOrSubprogram(index,parentTag,entryName,entryType,emptyRecord):
    """
    :param str entryType: either 'types' or 'subprograms'
    """
    global LOG_PREFIX
    utils.logging.logEnterFunction(LOG_PREFIX,"__searchIndexForTypeOrSubprogram",\
      {"parentTag":parentTag,"entryName":entryName,"entryType":entryType})

    result = None
    
    # create/lookup scope
    scope = __createScope(index,parentTag)
    # reverse access such that entries from the inner-most scope come first
    scopeEntities = reversed(scope[entryType])

    result = next((entry for entry in scopeEntities if entry["name"] == entryName),None)  
    if result is None:
        result         = emptyRecord
        result["name"] = entryName
        msg = "no entry found for {} '{}'.".format(entryType,entryName)
        if ERROR_HANDLING  == "strict":
            utils.logging.logError(LOG_PREFIX,"__searchIndexForTypeOrSubprogram",msg) 
            sys.exit(ERR_SCOPER_LOOKUP_FAILED)
        else:
            utils.logging.logWarning(LOG_PREFIX,"__searchIndexForTypeOrSubprogram",msg) 
        return result, False
    else:
        utils.logging.logDebug2(LOG_PREFIX,"__searchIndexForTypeOrSubprogram",\
          "entry found for {} '{}'".format(entryType,entryName)) 
        
        utils.logging.logLeaveFunction(LOG_PREFIX,"__searchIndexForTypeOrSubprogram")
        return result, True

# API
def searchIndexForVariable(index,parentTag,variableTag):
    """
    :param str parentTag: tag created of colon-separated identifiers, e.g. "mymodule" or "mymodule:mysubroutine".
    %param str variableTag% a simple identifier such as 'a' or 'A_d' or a more complicated tag representing a derived-type member, e.g. 'a%b%c'. Note that all array indexing expressions must be stripped away.
    :see: translator.translator.createIndexSearchTagForVariable(variableExpression)
    """
    global LOG_PREFIX
    utils.logging.logEnterFunction(LOG_PREFIX,"searchIndexForVariable",\
      {"parentTag":parentTag,"variableTag":variableTag})

    result = None
    # create/lookup scope
    scope = __createScope(index,parentTag)
    # reverse access such that entries from the inner-most scope come first
    scopeTypes = reversed(scope["types"])

    listOfVarNames = variableTag.split("%") 
    def lookupFromLeftToRight_(scopeVariables,pos=0):
        """
        :note: recursive
        """
        nonlocal scopeTypes
        nonlocal listOfVarNames
     
        varName = listOfVarNames[pos]
        if pos == len(listOfVarNames)-1:
            result = next((var for var in scopeVariables if var["name"] == varName),None)  
        else:
            try:
                matchingTypeVar = next((var for var in scopeVariables if var["name"] == varName),None)
                matchingType    = next((typ for typ in scopeTypes if typ["name"] == matchingTypeVar["kind"]),None)
                result = lookupFromLeftToRight_(reversed(matchingType["variables"]),pos+1)
            except Exception as e:
                raise e
                result = None
        return result
    result = lookupFromLeftToRight_(reversed(scope["variables"]))
    
    if result is None:
        result         = EMPTY_VARIABLE
        result["name"] = variableTag
        msg = "no entry found for variable '{}'.".format(variableTag)
        if ERROR_HANDLING  == "strict":
            utils.logging.logError(LOG_PREFIX,"searchIndexForVariable",msg) 
            sys.exit(ERR_SCOPER_LOOKUP_FAILED)
        else:
            utils.logging.logWarning(LOG_PREFIX,"searchIndexForVariable",msg) 
        return result, False
    else:
        utils.logging.logDebug2(LOG_PREFIX,"searchIndexForVariable",\
          "entry found for variable '{}'".format(variableTag)) 
        utils.logging.logLeaveFunction(LOG_PREFIX,"searchIndexForVariable")
        return result, True

def searchIndexForType(index,parentTag,typeName):
    """
    :param str parentTag: tag created of colon-separated identifiers, e.g. "mymodule" or "mymodule:mysubroutine".
    :param str typeName: lower case name of the searched type. Simple identifier such as 'mytype'.
    """
    utils.logging.logEnterFunction(LOG_PREFIX,"searchIndexForType",\
      {"parentTag":parentTag,"typeName":typeName})
    result = __searchIndexForTypeOrSubprogram(index,parentTag,typeName,"types",EMPTY_TYPE)
    utils.logging.logLeaveFunction(LOG_PREFIX,"searchIndexForType")
    return result

def searchIndexForSubprogram(index,parentTag,subprogramName):
    """
    :param str parentTag: tag created of colon-separated identifiers, e.g. "mymodule" or "mymodule:mysubroutine".
    :param str subprogramName: lower case name of the searched subprogram. Simple identifier such as 'mysubroutine'.
    """
    utils.logging.logEnterFunction(LOG_PREFIX,"searchIndexForSubprogram",\
      {"parentTag":parentTag,"subprogramName":subprogramName})
    result =  __searchIndexForTypeOrSubprogram(index,parentTag,subprogramName,"subprograms",EMPTY_SUBPROGRAM)
    utils.logging.logLeaveFunction(LOG_PREFIX,"searchIndexForSubprogram")
    return result
            
def indexVariableIsOnDevice(indexVar):
    return indexVar["device"] == True or\
           indexVar["declareOnTarget"] in ["alloc","to","from","tofrom"]
