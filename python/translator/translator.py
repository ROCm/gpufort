# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
import os
import addtoplevelpath
import os,sys,traceback
import utils
import logging
import collections
import ast
import re

# recursive inclusion
import indexer.indexertools as indexertools
import pyparsingtools as pyparsingtools

#from grammar import *
CASELESS    = True
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

def convertDeviceSubroutine(fortranSnippet,index=[],maxRecursions=10):
    """
    :return: C snippet equivalent to the original Fortran snippet.
    
    """
    global KEYWORDS 

    remainingRecursions=0
    def convertDeviceSubroutineRecursively(fortranSnippet,recursionsToGo):
        nonlocal remainingRecursions
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
