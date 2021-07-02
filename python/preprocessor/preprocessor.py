import os,sys
import re

import addtoplevelpath
import pyparsing as pyp

import utils.logging

ERR_PREPROCESSOR_MACRO_DEFINITION_NOT_FOUND = 11001

preprocessorDir = os.path.dirname(__file__)
exec(open("{0}/grammar.py".format(preprocessorDir)).read())

# API


def preprocess(fortranFileLines,options):
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
                if stmt.startswith("#"): #macrosV
                    processed = False
                    for result,_,__ in pp_define.scanString(stmt,1):
                        newMacro = { "name": result.name, "args": result.args.asList(), "subst": result.subst }
                        macroStack.push(newMacro)
                        processed = True
                    if not processed:
                        for result,_,__ in pp_define.scanString(stmt,1):
                           newMacro = { "name": result.name, "args": result.args.asList(), "subst": result.subst }
                           macroStack.push(newMacro)
                           found = True

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


def __expandMacros(text):
     oldResult = None
     result    = text
     # expand macro; one at a time
     while result != oldResult:
           oldResult = result
           result    = grammar.pp_value.transformString(result)
     return result

def __convertOperators(text):
    # replace C and Fortran operatos by python equivalents
    return grammar.pp_ops.transformString(text)

def __evaluateCondition(text):
    # replace C and Fortran operatos by python equivalents
    # TODO error handling
    return eval(__convertLinesToStatements(expandMacros_))

def __handlePreprocessorDirective_(statements,inActiveRegion,fortranFilepath,macroStack,regionStack):
    """
    :param str fortranFilepath: needed to load included files
    """
    includedRecords = []

    inActiveParentRegion = len(regionStack) < 2 or regionStack[-2] 
    inActiveRegion       = regionStack[-1]
    if inActiveRegion:
       # will be ignored if not in active region -> no descend -> problem, we do not know where we are
       handled = False
       # strip away whitespace chars
       strippedStmt = stmt.lstrip("# \t").lower()
       if strippedStmt.startswith("define"):`
           result = pp_dir_define.parseString(stmt,parseAll=True):
           newMacro = { "name": result.name, "args": result.args.asList(), "subst": result.subst }
           macroStack.push(newMacro)
           handled = True
       elif strippedStmt.startswith("undef"):
           result = pp_dir_define.parseString(stmt,parseAll=True):
           for macro in list(macroStack):
               if macro["name"] == result.name:
                   macroStack.remove(macro)
           handled = True
       elif strippedStmt.startswith("include"):
           result = pp_dir_include.parseString(stmt,parseAll=True):
           includeFileAbsPath = fortranFilepath + "/" + result.filename
           includedRecords = __preprocessAndNormalizeFortranFile(includeFileAbsPath,macroStack,regionStack):
           handled = True
       # if cond. true, push new region to stack
       # if parent not in active region, should also not be active
    # if and short hands
    if strippedStmt.startswith("ifdef"):
        if inActiveRegion:
            result = pp_dir_ifdef.parseString(stmt,parseAll=True):
            condition = "defined("+result.name+")"
            regionStack.append(
        else:
            regionStack.append(False)
    elif strippedStmt.startswith("ifndef"):
        if inActiveRegion:
            regionStack.append(condition)
        else:
            regionStack.append(False)
    elif strippedStmt.startswith("if"):
        if inActiveRegion:
            regionStack.append(condition)
        else:
            regionStack.append(False)
    elif strippedStmt.startswith("elif"):
        if inActiveParentRegion:
            pass
        else:
            regionStack.append(False)
    elif strippedStmt.startswith("else"):
        if inActiveParentRegion:
            pass
        else:
            regionStack.append(False)
    elif strippedStmt.startswith("endif"):
        regionStack.pop()
          # if parent not in active region, should also not be active

#def __applyMacros(statements,macroStack)
#    # TODO apply defined first; only if directive
#    
#
#    for macro in reversed(macroStack):
#        key   = macro["name"]
#        subst = macro["subst"] # keep the '\' here
#        # TODO move into macro creation and just read here
#        if not len(macro["args"]):
#            macro_pattern = pyp.Regex(r"\b"+key+r"\b",re.IGNORECASE).setParseAction(lambda tk: subst)
#        else: 
#            
#            for stmt in unrolledStatements:
#    
#    def expandMacros_(original):
#        oldResult = None
#        result    = original
#        # expand macro; one at a time
#        while result != oldResult:
#              oldResult = result
#              result    = grammar.pp_value.transformString(result)
#        # replace C and Fortran operatos by python equivalents
#        return result
#    def convertOperators_(text):
#        return grammar.pp_ops.transformString(text)
                        
def __convertLinesToStatements(lines):
    """
    Fortran lines can contain multiple statements that
    are separated by a semi-colon.
    This routine unrolls such lines into multiple single statements.
    Additionally, it converts single-line Fortran if statements
    into multi-line if-then-endif statements.
    """
    pContinuation          = re.compile(r"([\&]\s*\n)|(\n[!c\*]\$\w+\&)")
    # we look for a sequence ") <word>" were word != "then".
    pSingleLineIf          = re.compile("^(?P<indent>[\s\t]*)(?P<head>if\s*\(.+\))\s*\b(?!then)(?<body>\w.+)",re.IGNORECASE)
    
    # Try to determine indent char and width
    firstLine = lines[0]
    numIndentChars = len(lineStart)-len(lineStarts.lstrip(' '))
    if numIndentChars == 0 and defaultIndentChar == '\t':
        indentChar      = '\t'
        numIndentChars  = len(lineStart)-len(lineStarts.lstrip('\t'))
        indentIncrement = indentChar * indentWidthTab
    else:
        indentChar      = ' '
        indentIncrement = indentChar * indentWidthWhitespace
    indentOffset = numIndentChars * indentChar

    # make lower case, replace line continuation by whitespace, split at ";"
    singleLineStatements = pContinuation.sub(" "," ".join(lines)).split(";")
    # unroll single-line if
    unrolledStatements = []
    for stmt in singleLineStatements:
        match = pSingleLineIf.search(stmt)
        if match:
            if match.group("head").startswith("if"):
                THEN  = "then" 
                ENDIF = "endif"
            else:
                THEN  = "THEN"
                ENDIF = "ENDIF"
            unrolledStatements.append(match.group("indent") + match.group("head") + THEN + "\n")
            unrolledStatements.append(match.group("indent") + indentIncrement + match.group("body")) # line break is part of body
            unrolledStatements.append(match.group("indent") + ENDIF + "\n")
        else:
            unrolledStatements.append(indexOffset + stmt.lstrip(indentChar))
    return unrolledStatements

def __detectLineStarts(lines):
    """
    Fortran statements can be broken into multiple lines 
    via the '&'. This routine records in which line a statement
    (or multiple statements per line) begins.
    The difference between the line numbers of consecutive entries
    is the number of lines the first statement occupies.
    """
    pDirectiveContinuation = re.compile(r"\n[!c\*]\$\w+\&")

    # 1. save multi-line statements (&) in buffer
    buffering  = False
    lineStarts = []
    for lineno,line in enumerate(lines):
        # Continue buffering if multiline CUF/ACC/OMP statement
        buffering |= pDirectiveContinuation.match(line) != None
        if not buffering:
            lineStarts.append(lineno)
        if line.rstrip()[-1] in ['&','\\']:
            buffering = True
        else:
            buffering = False
    lineStarts.append(len(lines))
    return lineStarts

def __preprocessAndNormalize(fortranFileLines,options,fortranFilepath):
    """
    :param int includeLineno: Overwrite 
    """

    global LOG_PREFIX

    utils.logging.logEnterFunction(LOG_PREFIX,"preprocessAndNormalize",{
      "fortranFileLines":fortranFileLines,
      "fortranFilepath":fortranFilepath,
      "defaultIndentChar":defaultIndentChar,
      "indentWidthWhitespace":indentWidthWhitespace,
      "indentWidthTab":indentWidthTab
    })
    
    """
    :param list fileLines: Lines of a file, terminated with line break characters ('\n').
    :returns: a list of dicts with keys 'lineno', 'originalLines', 'statements'.
    """
    assert defaultIndentChar in [' ','\t'], "Indent char must be whitespace ' ' or tab '\\t'"

    lineStarts = __detectLineStarts(fortranFileLines)

    inActiveRegion = True

    # 2. now go through the blocks of buffered lines
    for i,_ in enumerate(lineStarts[:-1]):
        lineStart     = lineStarts[i]
        nextLineStart = lineStarts[i+1]
        lines         = lines[lineStart:nextLineStart]

        includedRecords = []
        isPreprocessorDirective = lines[0].startswith("#")
        if isPreprocessorDirective:
            unrolledStatements = list(lines)
            try:
                inActiveRegion, includedRecords = __handlePreprocessorDirective(lines,inActiveRegion,macroStack,regionStack)
                if not len(includedRecords):
                    __applyMacros
            except Exception as e:
                raise e
        else:
            if inActiveRegion:
                unrolledStatements1 = __convertToStatements(lines)
                "modified" = __applyMacros
                # todo apply macros to statements and convert to statements again if necessary
                unrolledStatements  = []

                record = {
                  "lineno":        lineno, # key
                  "lines":         lines,
                  "statements":    unrolledStatements,
                  "file":          fortranFilepath
                  "modified":      False
                  "included":      includedRecords
                }
                result.append(record)
    
    utils.logging.logLeaveFunction(LOG_PREFIX,"preprocessAndNormalize")
    return result


def __preprocessAndNormalizeFortranFile(fortranFilepath,macroStack=[],regionStack=[]):
    """
    :throws: IOError if the specified file cannot be found/accessed.
    """
    utils.logging.logEnterFunction(LOG_PREFIX,"__preprocessAndNormalizeFortranFile",{
      "fortranFilepath":fortranFilepath,
      "options":options,
      "defaultIndentChar":defaultIndentChar,
      "indentWidthWhitespace":indentWidthWhitespace,
      "indentWidthTab":indentWidthTab
    })

    try:
        with open(fortranFilepath,"r"):
            result = __preprocessAndNormalize(fortranFileLines,fortranFilepath,macroStack,regionStack)
            utils.logging.logLeaveFunction(LOG_PREFIX,"__preprocessAndNormalizeFortranFile")
            return result
    except Exception as e:
            raise e

def __initMacros(options):
    # init macro stack from compiler options
    macroStack = []
    for result,_,__ in pp_compiler_option.scanString(options):
        macro = { "name": result.name, "args": [], "subst": result.value },
        macroStack.append(macro)
    return macroStack

# API

def preprocessAndNormalizeFortranFile(fortranFilepath,options):
    """
    A C and Fortran preprocessor (cpp and fpp).

    Current limitations:

    * Only considers if/ifdef/ifndef/elif/else/define/undef/include 

    :return: The input data structure without the entries whose statements where not in active code region
             as determined by the preprocessor.
    :param str options: a sequence of compiler options such as '-D<key> -D<key>=<value>'.
    :throws: IOError if the specified file cannot be found/accessed.
    """
    global LOG_PREFIX
    global ERROR_HANDLING

    utils.logging.logEnterFunction(LOG_PREFIX,"preprocessAndNormalizeFortranFile",{
      "fortranFilepath":fortranFilepath,
      "options":options,
      "defaultIndentChar":defaultIndentChar,
      "indentWidthWhitespace":indentWidthWhitespace,
      "indentWidthTab":indentWidthTab
    })

    macroStack = __initMacros(options)

    # parse actions parameterized with macroStack
    def defined_(tokens):
        nonlocal macroStack
        if next((macro for macro in macroStack if macro["name"] == tokens[0]),None):
            return "1"
        else:
            return "0"
    def substitute_(tokens):
        name = tokens[0]
        args = tokens[1]
        nonlocal macroStack
        macro = next((macro for macro in reversed(macroStack) if macro["name"] == name),None)
        if macro:
            subst = macro["subst"]
            for n,placeholder in enumerate(macro["args"]):
                subst = re.sub(r"\b{}\b".format(placeholder),args[n],subst)
            return subst
        else:
            if ERROR_HANDLING=="strict":
                utils.logging.logError(LOG_PREFIX,"__preprocessAndNormalizeFortranFile","expanding macro failed: macro '{}' is not defined".format(name)))
                sys.abort(ERR_PREPROCESSOR_MACRO_DEFINITION_NOT_FOUND)
            else:
                utils.logging.logWarning(LOG_PREFIX,"__preprocessAndNormalizeFortranFile","expanding macro failed: macro '{}' is not defined".format(name)))
    
    grammar.pp_defined.setParseAction(defined_)
    grammar.pp_macro_eval.setParseAction(substitute_) 

    try:
        result = __preprocessAndNormalizeFortranFile(fortranFilepath,__initMacros(options))
        utils.logging.logLeaveFunction(LOG_PREFIX,"preprocessAndNormalizeFortranFile")
        return result
    except Exception as e:
        raise e

#def __applyMacros(statements,macroStack)
#    # TODO apply defined first; only if directive
#    
#
#    for macro in reversed(macroStack):
#        key   = macro["name"]
#        subst = macro["subst"] # keep the '\' here
#        # TODO move into macro creation and just read here
#        if not len(macro["args"]):
#            macro_pattern = pyp.Regex(r"\b"+key+r"\b",re.IGNORECASE).setParseAction(lambda tk: subst)
#        else: 
#            
#            for stmt in unrolledStatements:
#    
#    def expandMacros_(original):
#        oldResult = None
#        result    = original
#        # expand macro; one at a time
#        while result != oldResult:
#              oldResult = result
#              result    = grammar.pp_value.transformString(result)
#        # replace C and Fortran operatos by python equivalents
#        return result
#    def convertOperators_(text):
#        return grammar.pp_ops.transformString(text)
