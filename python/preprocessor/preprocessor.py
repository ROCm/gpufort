import os,sys
import re

import addtoplevelpath
import pyparsing as pyp

import utils.logging

ERR_PREPROCESSOR_MACRO_DEFINITION_NOT_FOUND = 11001

preprocessorDir = os.path.dirname(__file__)
exec(open("{0}/preprocessor_options.py.in".format(preprocessorDir)).read())
exec(open("{0}/grammar.py".format(preprocessorDir)).read())


def __expandMacros(text,macroStack):
    oldResult = ""
    result    = text
    # expand macro; one at a time
    macroNames = [ macro["name"] for macro in reversed(macroStack) ]
    macroName = "initial"
    while macroName:
        substring = ""
        macroName = None
        args      = []
        # finds all identifiers
        for match,start,end in pp_macro_eval.scanString(result):
            substring = result[start:end].strip(" \t\n")
            for name in macroNames:
                if substring.startswith(name):
                    macroName = name
                    args = match.args.asList()
                    break
            if macroName:
                break
        if macroName:
            macro = next((macro for macro in macroStack if macro["name"] == macroName),None)
            subst = macro["subst"].strip(" \n\t")
            for n,placeholder in enumerate(macro["args"]):
                subst = re.sub(r"\b{}\b".format(placeholder),args[n],subst)
            result = result.replace(substring,subst)
    return result

def __evaluateCondition(text,macroStack):
    # replace C and Fortran operators by python equivalents
    # TODO error handling
    return eval(pp_ops.transformString(pp_defined.transformString(__expandMacros(text,macroStack))))

def __handlePreprocessorDirective(lines,fortranFilepath,macroStack,regionStack1,regionStack2):
    """
    :param str fortranFilepath: needed to load included files where only relative path is specified
    :param list macroStack: A stack for storing/removing macro definition based on preprocessor directives.
    :param list regionStack1: A stack that stores if the current code region is active or inactive.
    :param list regionStack2: A stack that stores if any if/elif branch in the current if-elif-else-then
                              construct was active. This info is needed to decide if an else region 
                              should be activated.
    """
    global LOG_PREFIX

    includedRecords = []

    inActiveRegion  = regionStack1[-1]
    handled = False
    
    # strip away whitespace chars
    try:
        strippedFirstLine = lines[0].lstrip("# \t").lower()
        singleLineStatement = __convertLinesToStatements(lines)[0] # does not make sense for define
        if inActiveRegion:
           if strippedFirstLine.startswith("define"):
               # TODO error handling
               result   = pp_dir_define.parseString(lines[0],parseAll=True)
               substLines = [line.strip("\n")+"\n" for line in [result.subst] + lines[1:]]
               subst    = "".join(substLines).replace(r"\\","")
               newMacro = { "name": result.name, "args": result.args.asList(), "subst": subst }
               macroStack.append(newMacro)
               print(macroStack)
               handled = True
           elif strippedFirstLine.startswith("undef"):
               result = pp_dir_define.parseString(singleLineStatement,parseAll=True)
               for macro in list(macroStack):
                   if macro["name"] == result.name:
                       macroStack.remove(macro)
               handled = True
           elif strippedFirstLine.startswith("include"):
               result = pp_dir_include.parseString(singleLineStatement,parseAll=True)
               filename = result.filename.strip(" \t")
               if not filename.startswith("/"):
                   filename = fortranFilepath + "/" + filename
               includedRecords = __preprocessAndNormalizeFortranFile(filename,macroStack,regionStack1,regionStack2)
               handled = True
        # if cond. true, push new region to stack
        if strippedFirstLine.startswith("if"):
            if inActiveRegion and strippedFirstLine.startswith("ifdef"):
                result = pp_dir_ifdef.parseString(singleLineStatement,parseAll=True)
                condition = "defined("+result.name+")"
            elif inActiveRegion and strippedFirstLine.startswith("ifndef"):
                result = pp_dir_ifndef.parseString(singleLineStatement,parseAll=True)
                condition = "!defined("+result.name+")"
            elif inActiveRegion: # and strippedFirstLine.startswith("if"):
                result    = pp_dir_if.parseString(singleLineStatement,parseAll=True)
                condition = result.condition
            else:
                condition = "0"
            active = __evaluateCondition(condition,macroStack)
            regionStack1.append(active)
            anyIfElifActive = active
            regionStack2.append(anyIfElifActive)
            handled = True
        # elif
        elif strippedFirstLine.startswith("elif"):
            regionStack1.pop() # rm previous if/elif-branch
            if inActiveRegion and not regionStack2[-1]: # TODO allow to have multiple options specified at once
                result    = pp_dir_elif.parseString(singleLineStatement,parseAll=True)
                condition = result.condition
            else:
                condition = "0"
            active = __evaluateCondition(condition,macroStack)
            regionStack1.append(active)
            regionStack2[-1] = regionStack2[-1] or active
            handled = True
        # else
        elif strippedFirstLine.startswith("else"):
            regionStack1.pop() # rm previous if/elif-branch
            active = inActiveRegion and not regionStack2[-1]
            regionStack1.append(active)
            handled = True
        # endif
        elif strippedFirstLine.startswith("endif"):
            regionStack1.pop()
            regionStack2.pop()
            handled = True
    except Exception as e:
        raise e

    if not handled:
        # TODO add ignore filter
        utils.logging.logWarning(LOG_PREFIX,"__handlePreprocessorDirective",\
          "preprocessor directive '{}' was ignored".format(singleLineStatement))

    return includedRecords

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
    pSingleLineIf          = re.compile("^(?P<indent>[\s\t]*)(?P<head>if\s*\(.+\))\s*\b(?!then)(?P<body>\w.+)",re.IGNORECASE)
    
    # Try to determine indent char and width
    firstLine = lines[0]
    numIndentChars = len(firstLine)-len(firstLine.lstrip(' '))
    if numIndentChars == 0 and DEFAULT_INDENT_CHAR == '\t':
        indentChar      = '\t'
        numIndentChars  = len(firstLine)-len(firstLine.lstrip('\t'))
        indentIncrement = indentChar * INDENT_WIDTH_TABS
    else:
        indentChar      = ' '
        indentIncrement = indentChar * INDENT_WIDTH_WHITESPACE
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
            unrolledStatements.append(indentOffset + stmt.lstrip(indentChar))
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
        strippedLine = line.rstrip(" \t")  
        if len(strippedLine) and strippedLine[-1] in ['&','\\']:
            buffering = True
        else:
            buffering = False
    lineStarts.append(len(lines))
    return lineStarts

def __preprocessAndNormalize(fortranFileLines,fortranFilepath,macroStack,regionStack1,regionStack2):
    """
    :param list fileLines: Lines of a file, terminated with line break characters ('\n').
    :returns: a list of dicts with keys 'lineno', 'originalLines', 'statements'.
    """
    global LOG_PREFIX
    global ERROR_HANDLING
           
    global INDENT_WIDTH_WHITESPACE
    global INDENT_WIDTH_TABS
           
    global DEFAULT_INDENT_CHAR

    utils.logging.logEnterFunction(LOG_PREFIX,"preprocessAndNormalize",{
      "fortranFileLines":fortranFileLines,
      "fortranFilepath":fortranFilepath
    })
    assert DEFAULT_INDENT_CHAR in [' ','\t'], "Indent char must be whitespace ' ' or tab '\\t'"

    # 1. detect line starts
    lineStarts = __detectLineStarts(fortranFileLines)

    # 2. go through the blocks of buffered lines
    result = []
    for i,_ in enumerate(lineStarts[:-1]):
        lineStart     = lineStarts[i]
        nextLineStart = lineStarts[i+1]
        lines         = fortranFileLines[lineStart:nextLineStart]

        includedRecords = []
        isPreprocessorDirective = lines[0].startswith("#")
        if isPreprocessorDirective:
            try:
                includedRecords = __handlePreprocessorDirective(lines,fortranFilepath,macroStack,regionStack1,regionStack2)
            except Exception as e:
                raise e
        else:
            if regionStack1[-1]: # inActiveRegion
                # Convert line to statememts
                statements1 = __convertLinesToStatements(lines)
                # 2. Apply macros to statements
                print(statements1)

                statements2  = []
                for stmt1 in statements1:
                    statements2.append(__expandMacros(stmt1,macroStack))
                print(statements2)
                # 3. Above processing might introduce multiple statements per line againa.
                # Hence, convert each element of statements2 to single statements again
                statements3 = []
                for stmt2 in statements2:
                    for stmt3 in __convertLinesToStatements([stmt2]):
                        statements3.append(stmt3)

                record = {
                  "file":                fortranFilepath,
                  "lineno":              lineStart, # key
                  "lines":               lines,
                  "statements":          statements1,
                  "expandedStatements":  statements3,
                  "included":            includedRecords,
                  "modified":            False,
                }
                result.append(record)
    
    utils.logging.logLeaveFunction(LOG_PREFIX,"__preprocessAndNormalize")
    return result

def __preprocessAndNormalizeFortranFile(fortranFilepath,macroStack,regionStack1,regionStack2):
    """
    :throws: IOError if the specified file cannot be found/accessed.
    """
    utils.logging.logEnterFunction(LOG_PREFIX,"__preprocessAndNormalizeFortranFile",{
      "fortranFilepath":fortranFilepath
    })

    try:
        with open(fortranFilepath,"r") as infile:
            result = __preprocessAndNormalize(infile.readlines(),fortranFilepath,macroStack,regionStack1,regionStack2)
            utils.logging.logLeaveFunction(LOG_PREFIX,"__preprocessAndNormalizeFortranFile")
            return result
    except Exception as e:
            raise e

def __initMacros(options):
    # init macro stack from compiler options
    macroStack = []
    for result,_,__ in pp_compiler_option.scanString(options):
        macro = { "name": result.name, "args": [], "subst": result.value }
        macroStack.append(macro)
    return macroStack

# API

def preprocessAndNormalizeFortranFile(fortranFilepath,options=""):
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
      "options":options
    })

    macroStack = __initMacros(options)

    # parse actions parameterized with macroStack
    def defined_(tokens):
        nonlocal macroStack
        if next((macro for macro in macroStack if macro["name"] == tokens[0]),None):
            return "1"
        else:
            return "0"
    
    pp_defined.setParseAction(defined_)

    try:
        result = __preprocessAndNormalizeFortranFile(fortranFilepath,macroStack,\
           regionStack1=[True],regionStack2=[False]) # init value of regionStack[0] can be arbitrary
        utils.logging.logLeaveFunction(LOG_PREFIX,"preprocessAndNormalizeFortranFile")
        return result
    except Exception as e:
        raise e
