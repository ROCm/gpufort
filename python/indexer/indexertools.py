# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
#!/usr/bin/env python3
import addtoplevelpath
import os
import logging

# configurable parameters
indexerDir = os.path.dirname(__file__)
exec(open("{0}/indexertools_options.py.in".format(indexerDir)).read())

EMPTY = { "types" : [], "variables" : [] } 

EMPTY_VARIABLE = {                         
  "name"                       : "UNKNOWN",
  "fType"                      : "UNKNOWN",
  "kind"                       : "UNKNOWN",
  "bytesPerElement"            : "UNKNOWN",
  "cType"                      : "UNKNOWN",
  "fInterfaceType"             : "UNKNOWN",
  "fInterfaceQualifiers"       : "UNKNOWN",
  "parameter"                  : "UNKNOWN",
  "pointer"                    : "UNKNOWN",
  "device"                     : "UNKNOWN",
  "pinned"                     : "UNKNOWN",
  "managed"                    : "UNKNOWN",
  "allocatable"                : "UNKNOWN",
  "declaredOnTarget"           : "UNKNOWN",
  "rank"                       : -1,
  "unspecifiedBounds"          : "UNKNOWN",
  "lbounds"                    : "UNKNOWN",
  "counts"                     : "UNKNOWN",
  "totalCount"                 : "UNKNOWN",
  "totalBytes"                 : "UNKNOWN",
  "indexMacro"                 : "UNKNOWN",
  "indexMacroWithPlaceHolders" : "UNKNOWN"
}

def filterIndexByTag(index,tag,errorHandling=ERROR_HANDLING):
    """
    :return: only the structure(s) (module,program,subroutine,function) with
    a certain tag.
    """
    resultSet = [structure for structure in index if structure["tag"] == tag]
    if len(resultSet) != 1:
        msg = "'{}' entries found for tag '{}'. Expected to find a single entry.".format(len(resultSet),tag)
        if errorHandling == "strict":
            logging.getLogger("").error(msg)
            print("ERROR: "+msg,file=sys.stderr)
            sys.exit(1001)
        else:
            logging.getLogger("").warn(msg)
            return [ EMPTY ]
    else:
        msg = "'{}' entries found for tag '{}'".format(len(resultSet),tag)
        logging.getLogger("").debug2(msg)
        return resultSet

def searchIndexForVariable(index,variableExpression,errorHandling=ERROR_HANDLING):
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
        nonlocal errorHandling
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
        result         = EMPTY_VARIABLE
        result["name"] = variableExpression
        msg = "indexertools: no entry found for variable '{}'.".format(variableExpression)
        if errorHandling  == "strict":
            logging.getLogger("").error(msg)
            print("ERROR: "+msg,file=sys.stderr)
            sys.exit(1002)
        else:
            logging.getLogger("").warn(msg)
        return result, False
    else:
        msg = "indexertools: single entry found for variable '{}'".format(variableExpression)
        logging.getLogger("").debug2(msg)
        return result, True
