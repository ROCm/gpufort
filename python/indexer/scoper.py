# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
#!/usr/bin/env python3
import addtoplevelpath
import os
import logging
import copy

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

#def __addNewScope(newChildName,scopeContribution):
#    global currentNode
#    newChildTag = currentNode._tag + ":" + newChildName
#    new         = __Node(newChildTag,currentNode,scopeContribution)
#    currentNode = new
   
def __resolveDependencies(scope,indexRecord,index):
    """
    Include variable, type, and subprogram records from modules used
    by the current record (module,program or subprogram).

    :param dict scope: the scope that you updated with information from the used modules.
    :param dict indexRecord: a module/program/subprogram index record
    :param list index: list of module/program index records
    """
    for usedModule in indexRecord["usedModules"]:
        usedModuleFound = usedModule["name"] in MODULE_IGNORE_LIST
        # include definitions from other modules
        for module in index:
            if module["name"] == usedModule["name"]:
                usedModuleFound   = True
                includeAllEntries = not len(usedModule["only"])
                if includeAllEntries: # simple include
                    for entryType in __SCOPE_ENTRY_TYPES:
                        scope[entryType] += module[entryType]
                else:
                    for mapping in usedModule["only"]:
                        for entryType in __SCOPE_ENTRY_TYPES:
                            for entry in module[entryType]:
                                if entry["name"] == mapping["original"]:
                                    copiedEntry = entry.deepcopy(entry)
                                    copiedEntry["name"] = mapping["renaming"]
                                    scope[entryType].append(copiedEntry)
            if not usedModuleFound:
                msg = "no index record for module '{}' could be found".format(usedModule[name])
                if errorHandling == "strict":
                    utils.logError(msg) 
                    sys.exit(ERR_INDEXER_RESOLVE_DEPENDENCIES_FAILED)
                else:
                    utils.logWarn(msg)


def constructScope(index,tag,errorHandling=ERROR_HANDLING):
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
    
    # check if already a scope exists for the tag or if
    # it can be derived from a higher-level scope
    existingScope   = EMPTY_SCOPE
    nestingLevel    = -1 # -1 implies that nothing has been found
    scopesToDelete  = []
    for s in scopes:
        existingTag = s["tag"]
        if existingTag == tag[0:len(existingTag)]:
            existingScope = s
            nestingLevel  = len(existingTag.split(":"))-1
        else:
            scopesToDelete.append(s)
    # clean up scopes that are not used anymore 
    if REMOVE_OUTDATED_SCOPES:
        for s in scopesToDelete:
            scopes.remove(s)

    # return existing existingScope or construct it
    tagTokens = tag.split(":")
    if len(tagTokens)-1 == nestingLevel:
        return existingScope
    else:
        newScope = copy.deepcopy(existingScope)
       
        # we already have a scope for this record
        if nestingLevel >= 0:
            baseRecord = next((module for module in index if module["name"] == tagTokens[0]),None)  
            for l in range(1,nestingLevel+1):
                baseRecord = next((subprogram for subprogram in baseRecord["subprograms"] if subprogram["name"] == tagTokens[l]),None)
        if subprogram["name"] == name:
            currentRecordList = baseRecord["subprograms"]
        else:
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
                         scope[entryType] += currentRecord[entryType]
                    currentRecordList = currentRecord["subprograms"]
                    break
    SCOPES.append(newScope)
    return newScope

def searchIndexForVariable(index,parentTag,variableExpression,errorHandling=ERROR_HANDLING):
    """

    :param str variableExpression: a simple identifier such as 'a' or 'A_d' or a more complicated derived-type member expression such as 'a%b%c' or 'A%b(i)%c'.
    :param str parentTag: tag constructed of colon-separated identifiers, e.g. "mymodule" or "mymodule:mysubroutine".
                          This tag encodes the scope of the searched variable.
    :see: filterIndexByTag

    :note: Fortran does not support nested declarations of types. If a derived type
    has other derived type members, they must be declared before the definition of a new
    type that uses them.
    """
    result = None
    
    # construct/lookup scope
    scope = constructScope(index,parentTag,errorHandling)
    # reverse access such that entries from the inner-most scope come first

    def lookupFromLeftToRight_(scopeVariables,scopeTypes,expression):
        """
        :note: recursive
        """
        if "%" not in expression:
            result = next((var for var in scopeVariables if var["name"] == expression),None)  
        else:
            parts     = expression.split("%")
            typeVar   = parts[0].split("(")[0] # strip away array brackets
            remainder = "%".join(parts[1:])
            try:
                matchingTypeVar = next((var for var in scopeVariables if var["name"] == typeVar),None)
                matchingType    = next((typ for typ in scopeTypes if typ["name"] == matchingTypeVar["kind"]),None)
                result = lookupFromLeftToRight_(matchingType["variables"],matchingType["types"],remainder)
            except:
                result = None
        return result
    
    result = lookupFromLeftToRight_(reversed(scope["variables"]),reversed(scope["types"]),\
        variableExpression.lower().replace(" ",""))
    if result is None:
        result         = EMPTY_VARIABLE
        result["name"] = variableExpression
        msg = "scoper: no entry found for variable '{}'.".format(variableExpression)
        if errorHandling  == "strict":
            utils.logError(msg) 
            sys.exit(ERR_SCOPER_LOOKUP_FAILED)
        else:
            utils.logWarn(msg) 
        return result, False
    else:
        msg = "scoper: single entry found for variable '{}'".format(variableExpression)
        utils.logDebug2(msg) 
        return result, True
            
def indexVariableIsOnDevice(indexVar):
    return indexVar["device"] == True or\
           indexVar["declareOnTarget"] in ["alloc","to","from","tofrom"]

def __searchIndexForDerivedTypeOrSubprogram(index,parentTag,name,errorHandlin):
    scopeVariables   = reversed(scope["variables"])
    scopeTypes       = reversed(scope["types"])

    result = None
    
    # construct/lookup scope
    scope = constructScope(index,parentTag,errorHandling)
    # reverse access such that entries from the inner-most scope come first
    scopeTypes       = reversed(scope["types"])

    for structure in index:
        lookupFromLeftToRight(structure,variableExpression.lower().replace(" ",""))
    if result is None:
        result         = EMPTY_VARIABLE
        result["name"] = variableExpression
        msg = "scoper: no entry found for variable '{}'.".format(variableExpression)
        if errorHandling  == "strict":
            utils.logError(msg) 
            sys.exit(ERR_SCOPER_LOOKUP_FAILED)
        else:
            utils.logWarn(msg) 
        return result, False
    else:
        msg = "scoper: single entry found for variable '{}'".format(variableExpression)
        utils.logDebug2(msg) 
        return result, True
