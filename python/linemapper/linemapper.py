import os,sys
import re

import addtoplevelpath
import pyparsing as pyp

import utils.logging

ERR_LINEMAPPER_MACRO_DEFINITION_NOT_FOUND = 11001

linemapperDir = os.path.dirname(__file__)
exec(open("{0}/linemapper_options.py.in".format(linemapperDir)).read())
exec(open("{0}/grammar.py".format(linemapperDir)).read())

def __evaluateDefined(inputString,macroStack):
    # expand macro; one at a time
    result = inputString
    macroNames = [ macro["name"] for macro in reversed(macroStack) ]
    iterate = True
    while iterate:
        iterate   = False
        for match,start,end in pp_defined.scanString(result):
            substring = result[start:end].strip(" \t\n")
            subst = "1" if match.name in macroNames else "0"
            result = result.replace(substring,subst)
            iterate = True
            break
    return result

def __expandMacros(inputString,macroStack):
    # expand defined(...) expressions
    result = __evaluateDefined(inputString,macroStack)
    # expand macro; one at a time
    macroNames = [ macro["name"] for macro in macroStack ]
    iterate    = True
    while iterate:
        iterate = False
        # finds all identifiers
        for parseResult,start,end in pp_macro_eval.scanString(result):
            name = parseResult[0]
            if name in macroNames:
                substring = result[start:end].strip(" \t\n")
                args      = parseResult[1].asList()
                macro     = next((macro for macro in macroStack if macro["name"] == name),None)
                subst     = macro["subst"].strip(" \n\t")
                for n,placeholder in enumerate(macro["args"]):
                    subst = re.sub(r"\b{}\b".format(placeholder),args[n],subst)
                result  = result.replace(substring,subst)
                iterate = True
                break
    return result

def evaluateCondition(inputString,macroStack):
    """
    Evaluates preprocessor condition.
    :param str inputString: Expression as text.
    :note: Input validation performed according to:
           https://realpython.com/python-eval-function/#minimizing-the-security-issues-of-eval
    """
    transformedInputString = pp_ops.transformString(__expandMacros(inputString,macroStack))
    code = compile(transformedInputString, "<string>", "eval") 
    return eval(code, {"__builtins__": {}},{}) > 0

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
    
    def regionStackFormat(stack):
        return "-".join([str(int(el)) for el in stack])
    macroNames = ",".join([macro["name"] for macro in macroStack])
    
    utils.logging.logEnterFunction(LOG_PREFIX,"__handlePreprocessorDirective",\
      {"fortran-file-path": fortranFilepath,\
       "region-stack-1":    regionStackFormat(regionStack1),\
       "region-stack-2":    regionStackFormat(regionStack2),\
       "macro-names":       macroNames})

    includedRecords = []

    handled = False
    
    # strip away whitespace chars
    try:
        strippedFirstLine = lines[0].lstrip("# \t").lower()
        singleLineStatement = __convertLinesToStatements(lines)[0] # does not make sense for define
        if regionStack1[-1]:
           if strippedFirstLine.startswith("define"):
               utils.logging.logDebug3(LOG_PREFIX,"__handlePreprocessorDirective","found define in line '{}'".format(lines[0].rstrip("\n")))
               # TODO error handling
               result   = pp_dir_define.parseString(lines[0],parseAll=True)
               substLines = [line.strip("\n")+"\n" for line in [result.subst] + lines[1:]]
               subst    = "".join(substLines).replace(r"\\","")
               if result.args:
                   args = result.args.replace(" ","").split(",")
               else:
                   args = []
               newMacro = { "name": result.name, "args": args, "subst": subst }
               macroStack.append(newMacro)
               handled = True
           elif strippedFirstLine.startswith("undef"):
               utils.logging.logDebug3(LOG_PREFIX,"__handlePreprocessorDirective","found undef in line '{}'".format(lines[0].rstrip("\n")))
               result = pp_dir_define.parseString(singleLineStatement,parseAll=True)
               for macro in list(macroStack): # shallow copy
                   if macro["name"] == result.name:
                       macroStack.remove(macro)
               handled = True
           elif strippedFirstLine.startswith("include"):
               utils.logging.logDebug3(LOG_PREFIX,"__handlePreprocessorDirective","found include in line '{}'".format(lines[0].rstrip("\n")))
               result     = pp_dir_include.parseString(singleLineStatement,parseAll=True)
               filename   = result.filename.strip(" \t")
               currentDir = os.path.dirname(fortranFilepath)
               if not filename.startswith("/") and len(currentDir):
                   filename = os.path.dirname(fortranFilepath) + "/" + filename
               includedRecords = __preprocessAndNormalizeFortranFile(filename,macroStack,regionStack1,regionStack2)
               handled = True
        # if cond. true, push new region to stack
        if strippedFirstLine.startswith("if"):
            utils.logging.logDebug3(LOG_PREFIX,"__handlePreprocessorDirective","found if/ifdef/ifndef in line '{}'".format(lines[0].rstrip("\n")))
            if regionStack1[-1] and strippedFirstLine.startswith("ifdef"):
                result = pp_dir_ifdef.parseString(singleLineStatement,parseAll=True)
                condition = "defined("+result.name+")"
            elif regionStack1[-1] and strippedFirstLine.startswith("ifndef"):
                result = pp_dir_ifndef.parseString(singleLineStatement,parseAll=True)
                condition = "!defined("+result.name+")"
            elif regionStack1[-1]: # and strippedFirstLine.startswith("if"):
                result    = pp_dir_if.parseString(singleLineStatement,parseAll=True)
                condition = result.condition
            else:
                condition = "0"
            active = evaluateCondition(condition,macroStack)
            regionStack1.append(active)
            anyIfElifActive = active
            regionStack2.append(anyIfElifActive)
            handled = True
        # elif
        elif strippedFirstLine.startswith("elif"):
            utils.logging.logDebug3(LOG_PREFIX,"__handlePreprocessorDirective","found elif in line '{}'".format(lines[0].rstrip("\n")))
            regionStack1.pop() # rm previous if/elif-branch
            if regionStack1[-1] and not regionStack2[-1]: # TODO allow to have multiple options specified at once
                result    = pp_dir_elif.parseString(singleLineStatement,parseAll=True)
                condition = result.condition
            else:
                condition = "0"
            active = evaluateCondition(condition,macroStack)
            regionStack1.append(active)
            regionStack2[-1] = regionStack2[-1] or active
            handled = True
        # else
        elif strippedFirstLine.startswith("else"):
            utils.logging.logDebug3(LOG_PREFIX,"__handlePreprocessorDirective","found else in line '{}'".format(lines[0].rstrip("\n")))
            regionStack1.pop() # rm previous if/elif-branch
            active = regionStack1[-1] and not regionStack2[-1]
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
    global PATTERN_LINE_CONTINUATION

    pContinuation = re.compile(PATTERN_LINE_CONTINUATION)
    # we look for a sequence ") <word>" were word != "then".
    pSingleLineIf = re.compile(r"^(?P<indent>[\s\t]*)(?P<head>if\s*\(.+\))\s*\b(?!then)(?P<body>\w.+)",re.IGNORECASE)
    
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

    # replace line continuation by whitespace, split at ";"
    singleLineStatements = pContinuation.sub(" "," ".join(lines)).split(";")
    # unroll single-line if
    unrolledStatements = []
    for stmt in singleLineStatements:
        match = pSingleLineIf.search(stmt)
        if match:
            if match.group("head").startswith("if"):
                THEN  = " then" 
                ENDIF = "endif"
            else:
                THEN  = " THEN"
                ENDIF = "ENDIF"
            unrolledStatements.append(match.group("indent") + match.group("head") + THEN + "\n")
            unrolledStatements.append(match.group("indent") + indentIncrement + match.group("body").rstrip("\n")+"\n") # line break is part of body
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
    global ONLY_APPLY_USER_DEFINED_MACROS

    utils.logging.logEnterFunction(LOG_PREFIX,"preprocessAndNormalize",{
      "fortranFilepath":fortranFilepath
    })
    
    assert DEFAULT_INDENT_CHAR in [' ','\t'], "Indent char must be whitespace ' ' or tab '\\t'"

    # 1. detect line starts
    lineStarts = __detectLineStarts(fortranFileLines)

    # 2. go through the blocks of buffered lines
    records = []
    for i,_ in enumerate(lineStarts[:-1]):
        lineStart     = lineStarts[i]
        nextLineStart = lineStarts[i+1]
        lines         = fortranFileLines[lineStart:nextLineStart]

        includedRecords = []
        isPreprocessorDirective = lines[0].startswith("#")
        if isPreprocessorDirective and not ONLY_APPLY_USER_DEFINED_MACROS:
            try:
                includedRecords = __handlePreprocessorDirective(lines,fortranFilepath,macroStack,regionStack1,regionStack2)
                statements1 = []
                statements3 = []
            except Exception as e:
                raise e
        elif regionStack1[-1]: # inActiveRegion
                # Convert line to statememts
                statements1 = __convertLinesToStatements(lines)
                # 2. Apply macros to statements
                statements2  = []
                for stmt1 in statements1:
                    statements2.append(__expandMacros(stmt1,macroStack))
                # 3. Above processing might introduce multiple statements per line againa.
                # Hence, convert each element of statements2 to single statements again
                statements3 = []
                for stmt2 in statements2:
                    for stmt3 in __convertLinesToStatements([stmt2]):
                        statements3.append(stmt3)
    
        #if len(includedRecords) or (not isPreprocessorDirective and regionStack1[-1]):
        record = {
          "file":                    fortranFilepath,
          "lineno":                  lineStart, # key
          "lines":                   lines,
          "statements":              statements1,
          "expandedStatements":      statements3,
          "includedRecords":         includedRecords,
          "isPreprocessorDirective": isPreprocessorDirective,
          "isActive":                regionStack1[-1],
          "modified":                False
        }
        records.append(record)
    
    utils.logging.logLeaveFunction(LOG_PREFIX,"__preprocessAndNormalize")
    return records

def __preprocessAndNormalizeFortranFile(fortranFilepath,macroStack,regionStack1,regionStack2):
    """
    :throws: IOError if the specified file cannot be found/accessed.
    """
    utils.logging.logEnterFunction(LOG_PREFIX,"__preprocessAndNormalizeFortranFile",{
      "fortranFilepath":fortranFilepath
    })

    try:
        with open(fortranFilepath,"r") as infile:
            records = __preprocessAndNormalize(infile.readlines(),fortranFilepath,macroStack,regionStack1,regionStack2)
            utils.logging.logLeaveFunction(LOG_PREFIX,"__preprocessAndNormalizeFortranFile")
            return records
    except Exception as e:
            raise e

def __initMacros(options):
    """init macro stack from compiler options and user-prescribed config values."""
    global USER_DEFINED_MACROS

    macroStack = []
    macroStack += USER_DEFINED_MACROS
    for result,_,__ in pp_compiler_option.scanString(options):
        value = result.value
        if value == None:
            value == "1"
        macro = { "name": result.name, "args": [], "subst": result.value }
        macroStack.append(macro)
    return macroStack

def __groupModifiedRecords(records):
    global LINE_GROUPING_ENABLE
    global LINE_GROUPING_INCLUDE_BLANK_LINES
    """Find contiguous blocks of modified lines and blank lines between them."""
    EMPTY_BLOCK    = { "minLineno": -1, "maxLineno": -1, "orig": "", "subst": ""}
    blocks         = []
    currentRecords = []

    # auxiliary local functions
    def maxLineno_(record):
        record["lineno"] + len(record["lines"]) - 1
    def bordersPreviousRecord_(record):
        nonlocal currentRecords
        return i == 0 or record["lineno"] == maxLineno_(currentRecords[-1])+1
    def containsBlankLine_(record):
        return len(record["lines"]) == 1 and not len(record["lines"][0].lstrip(" \t\n"))
    def wasModified_(record):
        modified = record["modified"]
        for record in record["includedRecords"]:
            modified = modified or wasModified_(record)
        return modified
    def appendCurrentBlockIfNonEmpty_():
        # append current block if it is not empty
        # or does only contain blank lines.
        nonlocal blocks
        nonlocal currentRecords
        if len(currentRecords): # remove blank lines
            while containsBlankLine(currentRecords[-1]):
                currentRecords.pop()
        if len(currentRecords): # len might have changed
            block = dict(EMPTY_BLOCK) # shallow copy
            block["minLineno"] = currentRecords[0]["lineno"]
            block["maxLineno"] = maxLineno_(currentRecords[-1])

            def collectSubst_(record):
                if len(record["includedRecords"]):
                    for record in record["includedRecords"]:
                        return collectSubst_(record)
                elif record["modified"]:
                    return "\n".join([stmt.rstrip("\n") for rstmt in record["expandedStatements"]])
                else: # for included records
                    return "".join(record["lines"])

            for record in currentRecords:
                block["orig"]  += "".join(record["lines"])
                block["subst"] += collectSubst_(record)
            blocks.append(currentRecords)

    # 1. find contiguous blocks of modified or blank lines
    # 2. blocks must start with modified lines
    # 3. blank lines must be removed from tail of block
    for i,record in enumerate(records):
        lineno = record["lineno"]
        if wasModified_(record):
            if not LINE_GROUPING_ENABLE or not bordersPreviousRecord_(records):
                appendCurrentBlockIfNonEmpty_()
                currentRecords = []
            currentRecords.append(record)
        elif LINE_GROUPING_INCLUDE_BLANK_LINES and len(currentRecords) and containsBlankLine(record) and bordersPreviousRecord_(record):
            currentRecords.append(record)
    # append last block
    appendCurrentBlockIfNonEmpty_()
    return blocks

# API

def readFile(fortranFilepath,options=""):
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

    utils.logging.logEnterFunction(LOG_PREFIX,"readFile",{
      "fortranFilepath":fortranFilepath,
      "options":options
    })

    macroStack = __initMacros(options)
    try:
        records = __preprocessAndNormalizeFortranFile(fortranFilepath,macroStack,\
           regionStack1=[True],regionStack2=[True]) # init value of regionStack[0] can be arbitrary
        utils.logging.logLeaveFunction(LOG_PREFIX,"readFile")
        return records
    except Exception as e:
        raise e

def writeModifiedFile(infilePath,outfilePath,records):
    global FILE_MODIFICATION_WRAP_IN_IFDEF
    global FILE_MODIFICATION_MACRO

    blocks = __groupModifiedRecords(records)

    output      = ""
    blockId     = 0
    linesToSkip = -1
    with open(infilePath,"r") as infile:
        for lineno,line in enumerate(infile.readlines()):
            if lineno == blocks[blockId]["minLineno"]:
                block = blocks[blockId]
                linesToSkip = blocks["maxLineno"] - blockId["minLineno"] + 1
                subst = block["subst"]
                if FILE_MODIFICATION_WRAP_IN_IFDEF:
                    original = block["orig"]
                    output += "#ifdef {0}\n{1}\n#else\n{2}\n#endif\n".format(\
                      FILE_MODIFICATION_MACRO,result,original).split("\n")
                blockId +=1
            elif linesToSkip > 0:
                linesToSkip -= 1
            else:
                output += line
    with open(outfilePath,"w") as outfile:
        outfile.write(output.rstrip("\n"))

def renderFile(records,stage="expandedStatements",includeInactive=False,includePreprocessorDirectives=False):
    """
    :param str stage: either 'lines', 'statements' or 'expandedStatements', i.e. the preprocessor stage.
    :param bool includeInactive: include also code lines in inactive code regions.
    :param includePreprocessorDirectives: include also preprocessor directives in the output (except include directives).
    """
    utils.logging.logEnterFunction(LOG_PREFIX,"renderFile",\
      {"stage":stage})

    def renderFile_(records):
        nonlocal stage
        nonlocal includeInactive
        nonlocal includePreprocessorDirectives

        result = ""
        for record in records:
            condition1 = includeInactive or (record["isActive"])
            condition2 = includePreprocessorDirectives or (len(record["includedRecords"]) or not record["isPreprocessorDirective"])
            if condition1 and condition2:
                if len(record["includedRecords"]):
                    result += renderFile_(record["includedRecords"])
                else:
                    result += "".join(record[stage])
        return result

    utils.logging.logLeaveFunction(LOG_PREFIX,"renderFile")
    
    return renderFile_(records).strip("\n")
