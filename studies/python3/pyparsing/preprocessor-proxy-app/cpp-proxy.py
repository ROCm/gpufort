#!/usr/bin/env python3
import os,sys
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../python'))

import pyparsing as pyp

# grammar
pp_ident           = pyp.pyparsing_common.pp_identifier.copy()
LPAR,RPAR          = map(Suppress, "()")
pp_compiler_option = pyp.Regex(r"-D(?P<key>\w+)(=(?P<value>\"?\w+\"?))?")
# define / undef
#ex1: #define x  5
#ex1: #define x() 5
#ex1: #define x(a,b) 5*(a)*(b)
pp_define            = pyp.Regex(r"#\s*define\s*(?P<key>\w+)",re.IGNORECASE)
                       pyp.Optional(LPAR + Optional(pyp.delimitedList(pp_ident),default=[]) +RPAR,default=[]) + Regex(".+$")
pp_undef             = pyp.Regex(r"#\s*undef\s+(?P<key>\w+)",re.IGNORECASE)
# conditions
#pp_char             = pyp.Regex(r"'[ -~]'")
pp_number            = pyp.pyparsing_common.number.copy().setParseAction(lambda tk: str(tk))
pp_bool_true         = pyp.Regex(r".true.|true|1",re.IGNORECASE).setParseAction(lambda tk: 1)
pp_bool_false        = pyp.Regex(r".false.|false|0",re.IGNORECASE).setParseAction(lambda tk: 0)
pp_defined           = pyp.Regex(r"defined\s*\(\s*(?P<key>\w+)\s*\)",re.IGNORECASE)
pp_not_defined       = pyp.Regex(r"!\s*defined\s*\(\s*(?P<key>\w+)\s*\)",re.IGNORECASE)

pp_value       = Forward()
pp_arithm_expr =  pyp.infixNotation(pp_value, [
    (pyp.Literal('&&'), 2, pyp.opAssoc.LEFT),
    (pyp.Literal('||'), 2, pyp.opAssoc.LEFT),
    (pyp.Literal('!'), 1, pyp.opAssoc.LEFT),
])

pp_macro_eval   = pp_ident + LPAR + delimitedList( pp_arithm_expr ) + RPAR
pp_value      <<= ( pp_number | pp_bool_true | pp_bool_false | pp_ident | pp_macro_eval ) # | pp_char )

pp_comparison  = pp_arithm_expr + pyp.Regex(r"==|!=|<|>|<=|>=") + pp_arithm_expr

pp_op_and = pyp.Regex(r".and.|&&",re.IGNORECASE).setParseAction(lambda tk: "and" )
pp_op_or  = pyp.Regex(r".or.|\|\|",re.IGNORECASE).setParseAction(lambda tk: "or") 
pp_op_not = pyp.Regex(r".not.|!",re.IGNORECASE).setParseAction(lambda tk: "not")

pp_condition = pyp.infixNotation(pp_arithm_expr, [
    pp_op_and, 2, pyp.opAssoc.LEFT),
    pp_op_or, 2, pyp.opAssoc.LEFT),
    pp_op_not, 1, pyp.opAssoc.RIGHT),
])

# if
pp_ifdef           = pyp.Regex(r"#\s*ifdef\b(?P<key>\w+)",re.IGNORECASE)
pp_ifndef          = pyp.Regex(r"#\s*ifndef\s+(?P<key>\w+)",re.IGNORECASE)
pp_if              = pyp.Regex(r"#\s*if\s*(?P<condition>.+)$",re.IGNORECASE)
# elif
pp_if              = pyp.Regex(r"#\s*elif\s*(?P<condition>.+)",re.IGNORECASE)
# else
pp_else            = pyp.Regex(r"#\s*else\b",re.IGNORECASE).suppress()
# include 
pp_include         = pyp.Regex(r"#\s*include\s+(?P<filename>.+)",re.IGNORECASE)

# unsupported:
# pragmas
# ex: #pragma message ("KEY   is " KEY)
#pp_pragma_message  =  

# error
#pp_error           = pyp.Regex(r"#\s*error\s+(?P<message>.+)",re.IGNORECASE)


def __captureMultilinePreprocessorDirectives(lines):
    """
    Detects multiline statements such as
    """
    result = []
    buffering  = False
    lineStarts = []
    for lineno,line in enumerate(lines):
        if not buffering:
            lineStarts.append(lineno)
        if line.rstrip()[-1] in ["\\"]:
            buffering = True
        else:
            buffering = False
    lineStarts.append(len(lines))
    # 2. now go through the collapsed lines
    for i,_ in enumerate(lineStarts[:-1]):
        lineStart     = lineStarts[i]
        lineEnd       = lineStarts[i+1]
        originalLines = lines[lineStart:lineEnd]


def preprocess(fortranFileLines,optionsAsStr):
    """
    A C and Fortran preprocessor (cpp and fpp).

    Current limitations:

    * Only considers if/ifdef/ifndef/elif/else/define/undef/include 

    :param list normalizedFileLines: List of dictionaries with "statements" attribute (list of str).
                as produced by the normalizer. The statements must not contain any line breaks 
                or line continuation characters.
    }
    :return: The input data structure without the entries whose statements where not in active code region
             as determined by the preprocessor.
    """
    macroStack     = []
    regionStack    = [] 
    inActiveRegion = True

    preprocessedFilesLines = []
    for entry in normalizedFileLines:
        for statement in entry["statements"]:
            if inActiveRegion:
                if stmt.startswith("#"): #macros
                    for parseResult,_,__ in pp_define.scanString(stmt,1):
                        newMacro = { "name": parseResult[0].key, "args": parseResult[1], "subst": parseResult[2] }
                        macroStack.push(newMacro)
                        

                

# if inActiveRegion:
#     if encounter define:
#        add new macro
#     elif encounter undefine
#        remove macro
#     elif encounter if-branch: 
#         current <- push context to stack
#         inActiveRegionIfElifBranch <- 0
#         inActiveRegion <- evaluate if cond.
#         if inActiveRegion:
#             inActiveRegionIfElifBranch += 1
#         # active means we record lines
#     elif any other dexpression
#        
#     else
#        detect and apply macros to code line
# elif encounter elif-branch:
#     inActiveRegion <- evaluate if cond.
#     if inActiveRegion:
#         inActiveRegionIfElifBranch += 1
# elif encounter else-branch:
#     if inActiveRegionIfElifBranch == 0:
#         inActiveRegion <- True

#def scan(inputFilepath,optionsAsStr):
#    currentNode   = None
#    currentFile   = inputFilepath
#    currentLines  = []
#    currentLineno = -1
#    
#    # gpufort: map to STNode
#    class Node_:
#        def __init__(name):
#            nonlocal currentNode
#            nonlocal currentFile
#            nonlocal currentLines
#            nonlocal currentLineno
#            self._name      = name
#            self._parent    = currentNode
#            self._active    = True
#            self._file      = inputFilepath 
#            self._lines     = currentLines
#            self._lineno    = currentLineno
#            self._children  = []
#        def append_(self,child)
#            self._children.append(child)
#
#    class PPDefine_(Node_):
#        def __init__(name):
#            Node_.__init__(name)
#            self._args  = []
#            self._subst = ""
#    class PPUndefine_(Node_):
#        def __init__(name):
#            Node_.__init__(name)
#    
#    def findInTreeBottomUp_(node,filterFun,default=None):
#        def ascend(curr):
#            nonlocal filterFun
#            nonlocal default
#            if filterFun(curr):
#                return curr
#            elif self._parent == None:
#                return default
#            else: # not filterFun(curr) and self._parent != None
#                ascend(curr._parent)
#        return ascend(node)
#
#    def isDefined_(parentNode,symbolName):
#        """
#        :return: if the symbol with name 'symbolName' was defined before (and not undefined later).
#        """
#        def findLatestMention_(curr):
#            nonlocal symbolName
#            return type(curr) in PPUndefine_ and curr._name == symbolName or\
#                   type(curr) in PPDefine_ and curr._name == symbolName:
#        latestMention = findInTreeBottomUp_(node,findLatestMention_)
#        return type(latestMention) == PPDefine_
#
#    def inActiveBranch_(node,
#   
#    # am I currently in active block? no - then parse only PP directives, ignore the rest
#
#    # structure:
#    class PPIfBranchAndBlock_(Node_):
#        def __init__(name):
#            nonlocal currentNode
#            assert type(parent) == PPIfElseBlock_
#            Node_.__init__(name)
#            self._active = True
#        def isActive(self):
#            return self._condition > 0
#    class PPElifBranch_(Node_):
#        def 
#    class PPElseBranch_(Node_):
#        def 
#  
#        
#    def parseCompilerOptions_(optionsAsStr):
#        nonlocal currentNode
#        for parseResult in pp_compiler_option.searchString(optionsAsStr):
#            new = PPDefine_(parseResult.key)
#            new._text = 
#            currentNode.append(new)
#
#            print(parseResult.key)
#            print(parseResult.value)
#
#    def handleInclude_(inputFilepath):
#        pass
#
#    root = Root("root")
#    currentNode = root
#    def scanRecursively_(inputFilepath):
#        """
#        :throws: IOError. If the file could not opened.
#        """
#        nonlocal currentNode
#        nonlocal currentFile
#        nonlocal currentLines
#        nonlocal currentLineno
#        
#        try:
#            content = ""
#            with open(inputFilepath,"r") as infile:
#                normalizedLines = __normalize(infile.readlines())
#            print(content)
#        except IOError as ioError:
#            pass
#
#
#    parseCompilerOptions_(optionsAsStr)
#    scanRecursively_(inputFilepath)
#
#
#if __name__ == "__main__":
#    optionsAsStr = "-Dfoo1 -Dfoo2=bar2 -Dfoo3=\"bar3\""
#    scan("./snippet1.f90",optionsAsStr)
#    
