import os,sys
import re

# API

LOG_PREFIX="scanner.normalizer"

def normalize(fortranFileLines,fortranFilepath,defaultIndentChar=' ',indentWidthWhitespace=2,indentWidthTabs=1):
    global LOG_PREFIX

    utils.logging.logEnterFunction(LOG_PREFIX,"normalize",{
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
    pDirectiveContinuation = re.compile(r"\n[!c\*]\$\w+\&")
    pContinuation          = re.compile(r"([\\\&]\s*\n)|(\n[!c\*]\$\w+\&)")
    # we look for a sequence ") <word>" were word != "then".
    pSingleLineIf          = re.compile("^(?P<indent>[\s\t]*)(?P<head>if\s*\(.+\))\s*\b(?!then)(?<body>\w.+)",re.IGNORECASE)

    result = []
    # 1. collapse multi-line statements (&)
    buffering  = False
    lineStarts = []
    for lineno,line in enumerate(lines):
        # Continue buffering if multiline CUF/ACC/OMP statement
        buffering |= pDirectiveContinuation.match(line) != None
        if not buffering:
            lineStarts.append(lineno)
        if line.rstrip()[-1] in ['&',"\\"]:
            buffering = True
        else:
            buffering = False
    lineStarts.append(len(lines))
    # 2. now go through the collapsed lines
    for i,_ in enumerate(lineStarts[:-1]):
        lineStart     = lineStarts[i]
        lineEnd       = lineStarts[i+1]
        originalLines = lines[lineStart:lineEnd]

        # Try to determine indent char and width
        firstLine = originalLines[0]
        numIndentChars = len(lineStart)-len(lineStarts.lstrip(' '))
        if numIndentChars == 0 and defaultIndentChar == '\t':
            indentChar      = '\t'
            numIndentChars  = len(lineStart)-len(lineStarts.lstrip('\t'))
            indentIncrement = indentChar * indentWidthTab
        else:
            indentChar      = ' '
            indentIncrement = indentChar * indentWidthWhitespace

        # make lower case, replace line continuation by whitespace, split at ";"
        singleLineStatements = pContinuation.sub(" "," ".join(originalLines)).split(";")
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
                unrolledStatements.append(stmt)
       
        lineRecord = {
          "lineno": lineno,
          "statements": unrolledStatements,
          "originalLines": originalLines,
          "file": fortranFilepath
        }
        result.append(lineRecord)
    
    utils.logging.logLeaveFunction(LOG_PREFIX,"normalize")
    return result

def readFortranAndNormalize(fortranFilepath,defaultIndentChar=' ',indentWidthWhitespace=2,indentWidthTabs=1):
    """
    """
    utils.logging.logEnterFunction(LOG_PREFIX,"readFortranAndNormalize",{
      "fortranFilepath":fortranFilepath,
      "defaultIndentChar":defaultIndentChar,
      "indentWidthWhitespace":indentWidthWhitespace,
      "indentWidthTab":indentWidthTab
    })

    result = normalize(fortranFileLines,fortranFilepath,defaultIndentChar,indentWidthWhitespace,indentWidthTabs)
    
    utils.logging.logLeaveFunction(LOG_PREFIX,"readFortranAndNormalize")
    return result
