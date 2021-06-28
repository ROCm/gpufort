#!/usr/bin/env python3
import os,sys
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../python'))

import re
import pyparsing as pyp

EMPTY_MACRO = { "name": "", "args": [], "subst": []}

# grammar
ident              = pp.pyparsing_common.identifier
LPAR,RPAR          = map(Suppress, "()")
pp_compiler_option = pyp.Regex(r"-D(?P<key>\w+)(=(?P<value>\"?\w+\"?))?")
# define / undef
pp_define          = pyp.Regex(r"#\s*define",re.IGNORECASE).suppress() + ident +\
                       pyp.Optional(LPAR + pyp.delimitedList(ident),default=[] + RPAR) + Regex(".+$")
pp_undefine        = pyp.Regex(r"#\s*undefine\s+(?P<key>\w+)",re.IGNORECASE)
# if
pp_ifdef           = pyp.Regex(r"#\s*ifdef\b(?P<key>\w+)",re.IGNORECASE)
pp_ifndef          = pyp.Regex(r"#\s*ifndef\s+(?P<key>\w+)",re.IGNORECASE)
pp_if              = pyp.Regex(r"#\s*if\s+(?P<condition>.+)",re.IGNORECASE)
# elif
pp_elif            = pyp.Regex(r"#\s*elif\s+(?P<condition>.+)",re.IGNORECASE)
# else
pp_else            = pyp.Regex(r"#\s*else\b",re.IGNORECASE).suppress()
# include 
pp_include         = pyp.Regex(r"#\s*include\s+(?P<filename>.+)",re.IGNORECASE)
    
def scan(inputFilepath,optionsAsStr):
    currentNode   = None
    currentFile   = inputFilepath
    currentLines  = []
    currentLineno = -1
    
    # gpufort: map to STNode
    class Node_:
        def __init__(name):
            nonlocal currentNode
            nonlocal currentFile
            nonlocal currentLines
            nonlocal currentLineno
            self._name      = name
            self._parent    = currentNode
            self._active    = True
            self._file      = inputFilepath 
            self._lines     = currentLines
            self._lineno    = currentLineno
            self._children  = []
        def append_(self,child)
            self._children.append(child)

    class PPDefine_(Node_):
        def __init__(name):
            Node_.__init__(name)
            self._args  = []
            self._subst = ""
    class PPUndefine_(Node_):
        def __init__(name):
            Node_.__init__(name)
    
    def findInTreeBottomUp_(node,filterFun,default=None):
        def ascend(curr):
            nonlocal filterFun
            nonlocal default
            if filterFun(curr):
                return curr
            elif self._parent == None:
                return default
            else: # not filterFun(curr) and self._parent != None
                ascend(curr._parent)
        return ascend(node)

    def isDefined_(parentNode,symbolName):
        """
        :return: if the symbol with name 'symbolName' was defined before (and not undefined later).
        """
        def findLatestMention_(curr):
            nonlocal symbolName
            return type(curr) in PPUndefine_ and curr._name == symbolName or\
                   type(curr) in PPDefine_ and curr._name == symbolName:
        latestMention = findInTreeBottomUp_(node,findLatestMention_)
        return type(latestMention) == PPDefine_

    def inActiveBranch_(node,
   
    # am I currently in active block? no - then parse only PP directives, ignore the rest

    # structure:
    class PPIfElseBlock_(Node_):
        def __init__(expression):
            Node_.__init__(expression)
    class PPIfBranch_(Node_):
        def __init__(name):
            nonlocal currentNode
            assert type(parent) == PPIfElseBlock_
            Node_.__init__(name)
            self._active = True
        def isActive(self):
            return self._condition > 0
    class PPElifBranch_(Node_):
        def 
    class PPElseBranch_(Node_):
        def 
  
        
    def parseCompilerOptions_(optionsAsStr):
        nonlocal currentNode
        for parseResult in pp_compiler_option.searchString(optionsAsStr):
            new = PPDefine_(parseResult.key)
            new._text = 
            currentNode.append(new)

            print(parseResult.key)
            print(parseResult.value)

    def handleInclude_(inputFilepath):
        pass

    root = Root("root")
    currentNode = root
    def scanRecursively_(inputFilepath):
        """
        :throws: IOError. If the file could not opened.
        """
        nonlocal currentNode
        nonlocal currentFile
        nonlocal currentLines
        nonlocal currentLineno
        
        try:
            content = ""
            with open(inputFilepath,"r") as infile:
                normalizedLines = __normalize(infile.readlines())
            print(content)
        except IOError as ioError:
            pass


    parseCompilerOptions_(optionsAsStr)
    scanRecursively_(inputFilepath)


if __name__ == "__main__":
    optionsAsStr = "-Dfoo1 -Dfoo2=bar2 -Dfoo3=\"bar3\""
    scan("./snippet1.f90",optionsAsStr)
    
