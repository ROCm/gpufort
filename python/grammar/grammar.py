import os, sys

# third-party modules
from pyparsing import *

# local modules
from grammar.cudafor import *

# The module executes the content of this module can set
# the GRAMMAR_DIR global variable beforehand.
if not 'GRAMMAR_DIR' in globals():
    GRAMMAR_DIR = os.path.dirname(os.path.abspath(__file__))
if not 'CASELESS' in globals():
    CASELESS = False
if CASELESS == True:
    CASELESS_LITERAL = CaselessLiteral
else:
    CASELESS_LITERAL = Literal

# Performance Tips:
# - try using enablePackrat()
# - use MatchFirst(|) instead of Or(^)
ParserElement.setDefaultWhitespaceChars("\r\n\t &;")
ParserElement.enablePackrat() 

# helper functions
def makeCaselessLiteral(commaSeparatedList,suppress=False,forceCaseLess=False):
     if forceCaseLess:
        result1 = map(CaselessLiteral, commaSeparatedList.split(","))
     else: # can be overwritten via options
        result1 = map(CASELESS_LITERAL, commaSeparatedList.split(","))
     if suppress:
        result2 = []
        for element in result1:
             result2.append(element.suppress())
        return result2
     else:
        return result1

def separatedSequence(tokens,separator=Suppress(",")):
    result = tokens[0]
    for token in tokens[1:]:
         result += separator + token
    return result

exec(open(os.path.join(GRAMMAR_DIR, "grammar_f03.py.in")).read())
exec(open(os.path.join(GRAMMAR_DIR, "grammar_directives.py.in")).read())
exec(open(os.path.join(GRAMMAR_DIR, "grammar_cuf.py.in")).read())
exec(open(os.path.join(GRAMMAR_DIR, "grammar_acc.py.in")).read())
exec(open(os.path.join(GRAMMAR_DIR, "grammar_epilog.py.in")).read())
