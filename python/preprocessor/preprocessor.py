import os,sys
import re

import addtoplevelpath
import pyparsing as pyp

__LOG_PREFIX="preprocessor.preprocessor"

preprocessorDir = os.path.dirname(__file__)
exec(open("{0}/grammar.py".format(preprocessorDir)).read())

# API

def __convertToStatements(lines):
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

def __mapLinesToStatements(lines):
    pDirectiveContinuation = re.compile(r"\n[!c\*]\$\w+\&")

    result = []
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
                if stmt.startswith("#"): #macros
                    for parseResult,_,__ in pp_define.scanString(stmt,1):
                        newMacro = { "name": parseResult[0].key, "args": parseResult[1], "subst": parseResult[2] }
                        macroStack.push(newMacro)

def __handlePreprocessorDirective_(originalLines,macroStack,regionStack):
    return False, []

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

def __applyMacros(statements,macroStack)
    # TODO apply defined first; only if directive
    

    for macro in reversed(macroStack):
        key   = macro["name"]
        subst = macro["subst"] # keep the '\' here
        # TODO move into macro creation and just read here
        if not len(macro["args"]):
            macro_pattern = pyp.Regex(r"\b"+key+r"\b",re.IGNORECASE).setParseAction(lambda tk: subst)
        else: 
            
            for stmt in unrolledStatements:
                        


def preprocessAndNormalize(fortranFileLines,options,fortranFilepath,includeLineno=-1):
    """
    :param int includeLineno: Overwrite 
    """

    global __LOG_PREFIX

    utils.logging.logEnterFunction(__LOG_PREFIX,"preprocessAndNormalize",{
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

    lineStarts = __mapLinesToStatements(fortranFileLines)

    macroStack     = []
    regionStack    = []
    inActiveRegion = []

    # 2. now go through the blocks of buffered lines
    for i,_ in enumerate(lineStarts[:-1]):
        lineStart     = lineStarts[i]
        lineEnd       = lineStarts[i+1]
        originalLines = lines[lineStart:lineEnd]

        includedRecords = []
        isPreprocessorDirective = originalLines[0].startswith("#")
        if isPreprocessorDirective:
            unrolledStatements = list(originalLines)
            try:
                inActiveRegion, includedRecords = __handlePreprocessorDirective(originalLines,inActiveRegion,macroStack,regionStack)
                __applyMacros
            except Exception as e:
                raise e
        else:
            if inActiveRegion:
                unrolledStatements1 = __convertToStatements(originalLines)
                "modified" = __applyMacros
                # todo apply macros to statements and convert to statements again if necessary
                unrolledStatements  = []

        record = {
          "lineno":        lineno, # key
          "lines":         originalLines,
          "statements":    unrolledStatements,
          "file":          fortranFilepath
          "modified":      False
          "included":      includedRecords
        }
        result.append(record)
    
    utils.logging.logLeaveFunction(__LOG_PREFIX,"preprocessAndNormalize")
    return result


def __preprocessAndNormalizeFortranFile(fortranFilepath,macroStack=[],regionStack=[]):
    """
    :throws: IOError if the specified file cannot be found/accessed.
    """
    utils.logging.logEnterFunction(__LOG_PREFIX,"__preprocessAndNormalizeFortranFile",{
      "fortranFilepath":fortranFilepath,
      "options":options,
      "defaultIndentChar":defaultIndentChar,
      "indentWidthWhitespace":indentWidthWhitespace,
      "indentWidthTab":indentWidthTab
    })

    try:
        with open(fortranFilepath,"r"):
            result = __preprocessAndNormalize(fortranFileLines,fortranFilepath,options,macroStack,regionStack)
            utils.logging.logLeaveFunction(__LOG_PREFIX,"__preprocessAndNormalizeFortranFile")
            return result
    except Exception as e:
            raise e


# API

def preprocessAndNormalizeFortranFile(fortranFilepath,options):
    """
    :param str options: a sequence of compiler options such as '-D<key> -D<key>=<value>'.
    :throws: IOError if the specified file cannot be found/accessed.
    """
    utils.logging.logEnterFunction(__LOG_PREFIX,"preprocessAndNormalizeFortranFile",{
      "fortranFilepath":fortranFilepath,
      "options":options,
      "defaultIndentChar":defaultIndentChar,
      "indentWidthWhitespace":indentWidthWhitespace,
      "indentWidthTab":indentWidthTab
    })

    macroStack = []
    for match,_,__ in pp_compiler_option.scanString(options):
        if match.

    try:
        result = __preprocessAndNormalizeFortranFile(fortranFilepath,options)
        utils.logging.logLeaveFunction(__LOG_PREFIX,"preprocessAndNormalizeFortranFile")
        return result
    except Exception as e:
        raise e
