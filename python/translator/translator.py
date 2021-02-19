import os
import addtoplevelpath
import os,sys,traceback
import utils
import logging
import collections
import ast
import re

#from grammar import *
GRAMMAR_DIR = os.path.join(os.path.dirname(__file__),"../grammar")
exec(open("{0}/grammar.py".format(GRAMMAR_DIR)).read())

exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "translator_options.py.in")).read())
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "translator_base.py.in")).read())
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "translator_f03.py.in")).read())
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "translator_directives.py.in")).read())
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "translator_cuf.py.in")).read())
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "translator_acc.py.in")).read())
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "translator_codegen.py.in")).read())

# API

def convertArithmeticExpression(fortranSnippet):
    return ( matrixArithmeticExpression | complexArithmeticExpression | arithmeticExpression ).parseString(fortranSnippet)[0].cStr()

remainingRecursions=0
def convertCufLoopKernel2Hip(fortranSnippet,index=[],maxRecursions=10):
    """
    Return a csnippet equit
    """
    global KEYWORDS 

    def convertCufKernel2HipRecursively(fortranSnippet,recursionsToGo):
        global remainingRecursions
        remainingRecursions = recursionsToGo
        try:
            return cufLoopKernel.parseString(fortranSnippet)[0]
        except ParseBaseException as pbe:
            if recursionsToGo <= 0:
                raise pbe
            else:
                lineno = pbe.__getattr__("lineno")
                lines = fortranSnippet.split("\n")
                lines[lineno-1] = "! TODO could not parse: {}".format(lines[lineno-1])
                modifiedFortranSnippet = "\n".join(lines)
                #print(modifiedFortranSnippet)
                return convertCufKernel2HipRecursively(modifiedFortranSnippet,recursionsToGo-1)
        except Exception as e:
            raise e        
   
    fortranSnippet = prepareFortranSnippet(fortranSnippet)
    #print(fortranSnippet)
    try:
        result = convertCufKernel2HipRecursively(fortranSnippet,maxRecursions)
        cSnippet = utils.prettifyCCode(result.cStr())
        kernelLaunchInfo = result.kernelLaunchInfo()
        identifierNames  = result.allIdentifiers()
        reductionVars    = result.reductionVars()
        loopVars         = result.loopVars()
        #print(loopVars)
        localLvalues = list(filter(lambda x: x not in loopVars,result.allLocalLvalues())) 
        problemSize = result.problemSize()
        #print(remainingRecursions)
        if remainingRecursions < maxRecursions:
            body = "\n".join(fortranSnippet.split("\n")[1:])
            identifierNames = list(filter(lambda x: x.lower() not in KEYWORDS,[makeFStr(ident) for ident in identifier.searchString(body)]))
            #print(identifierNames) 
    except Exception as e:
        cSnippet = "" 
        pragmaLine = fortranSnippet.split("\n")[0]
        body = "\n".join(fortranSnippet.split("\n")[1:])
        kernelLaunchInfo = cufKernelDo.parseString(pragmaLine)[0] 
        numOuterLoopsToMap = int(kernelLaunchInfo._numOuterLoopsToMap)
        #print(body)
        identifierNames = list(filter(lambda x: x.lower() not in KEYWORDS,[makeFStr(ident) for ident in identifier.searchString(body)]))
        reductionVars   = []
        loopVars        = []
        localLvalues    = []
        problemSize = ["TODO unknown"]*numOuterLoopsToMap
        #print(type(e))
        raise(e)
    cSnippet = postprocessCSnippet(cSnippet)
    return cSnippet, problemSize, kernelLaunchInfo, identifierNames, localLvalues, loopVars

def convertDeviceSubroutine(fortranSnippet,index=[],maxRecursions=10):
    """
    :return: C snippet equivalent to the original Fortran snippet.
    
    """
    global KEYWORDS 

    def convertDeviceSubroutineRecursively(fortranSnippet,recursionsToGo):
        global remainingRecursions
        remainingRecursions = recursionsToGo
        try:
            return subroutine.parseString(fortranSnippet)[0]
        except ParseBaseException as pbe:
            if recursionsToGo <= 0:
                raise pbe
            else:
                lineno = pbe.__getattr__("lineno")
                lines = fortranSnippet.split("\n")
                lines[lineno-1] = "! TODO could not parse: {}".format(lines[lineno-1])
                modifiedFortranSnippet = "\n".join(lines)
                #print(modifiedFortranSnippet)
                return convertDeviceSubroutineRecursively(modifiedFortranSnippet,recursionsToGo-1)
        except Exception as e:
            raise e        
 
    fortranSnippet = prepareFortranSnippet(fortranSnippet)
    #print("prepared::::: "+fortranSnippet)
    try:
        result   = convertDeviceSubroutineRecursively(fortranSnippet,maxRecursions)
        cBody    = utils.prettifyCCode(result.bodyCStr())
        name     = result.nameCStr()
        argNames = result.argNamesCStr()
        #print(remainingRecursions)
    except Exception as e:
        cBody = ""
        name = None
        argNames = [] 
        #print(type(e))
        raise e        
    cBody = postprocessCSnippet(cBody)
    return name,argNames,cBody

remainingRecursions=0
def convertAccLoopKernel2Hip(fortranSnippet,index=[],maxRecursions=30):
    """
    Return a csnippet equivalent to the original Fortran code.
    """
    global KEYWORDS 

    def convertAccKernel2HipRecursively(fortranSnippet,recursionsToGo):
        global remainingRecursions
        remainingRecursions = recursionsToGo
        try:
            return accLoopKernel.parseString(fortranSnippet)[0]
        except ParseBaseException as pbe:
            if recursionsToGo <= 0:
                raise pbe
            else:
                lineno = pbe.__getattr__("lineno")
                lines = fortranSnippet.split("\n")
                lines[lineno-1] = "! TODO could not parse: {}".format(lines[lineno-1])
                modifiedFortranSnippet = "\n".join(lines)
                #print(modifiedFortranSnippet)
                return convertAccKernel2HipRecursively(modifiedFortranSnippet,recursionsToGo-1)
        except Exception as e:
            raise e        
   
    fortranSnippet = prepareFortranSnippet(fortranSnippet)
    try:
        result           = convertAccKernel2HipRecursively(fortranSnippet,maxRecursions)
        # TODO modify AST here
        cSnippet         = utils.prettifyCCode(result.cStr())
        
        kernelLaunchInfo = result.kernelLaunchInfo()
        identifierNames  = result.allIdentifiers()
        loopVars         = result.loopVars()
        localLvalues     = list(filter(lambda x: x not in loopVars, result.allLocalLvalues())) 
        localLvalues     += result.threadPrivateVars()
        reductionVars    = result.reductionVars()
        problemSize      = result.problemSize()
        #print(remainingRecursions)
        if remainingRecursions < maxRecursions:
            body = "\n".join(fortranSnippet.split("\n")[1:])
            identifierNames = list(set(filter(lambda x: x.lower() not in KEYWORDS,[makeFStr(ident) for ident in identifier.searchString(body)])))
            #print(identifierNames) 
        cSnippet = postprocessCSnippet(cSnippet)
    except Exception as e:
        raise e
        logger = logging.getLogger('') 
        logger.error("failed to convert kernel:\n{}".format(fortranSnippet))
        logger.error(str(e))
        cSnippet = "" 
        pragmaLine = fortranSnippet.split("\n")[0]
        body = "\n".join(fortranSnippet.split("\n")[1:])
        #kernelLaunchInfo = acc.parseString(pragmaLine)[0] 
        kernelLaunchInfo = LaunchInfo()
        #print(body)
        identifierNames = list(set(filter(lambda x: x.lower() not in KEYWORDS,[makeFStr(ident) for ident in identifier.searchString(body)])))
        numOuterLoopsToMap     = int(kernelLaunchInfo._numOuterLoopsToMap)
        loopVars        = []
        localLvalues    = []
        reductionVars   = []
        problemSize     = ["TODO unknown"]*numOuterLoopsToMap
        #print(type(e))
    return cSnippet, problemSize, kernelLaunchInfo, identifierNames, localLvalues, loopVars, reductionVars
