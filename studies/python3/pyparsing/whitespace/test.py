# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import pyparsing as pyp
import re

pyp.ParserElement.setDefaultWhitespaceChars(" \t")

TERM = pyp.Regex(r"\n|;|;[ \t]*\n?").suppress()

BLANK_LINE  = pyp.Regex(r"^[ \t]*$",re.IGNORECASE)
MODULE      = pyp.Regex(r"module",re.IGNORECASE)
END_MODULE  = pyp.Regex(r"end\s*module(\s*\w+)?",re.IGNORECASE)

identifier = pyp.pyparsing_common.identifier

stmt_module_begin = MODULE + pyp.pyparsing_common.identifier + TERM
stmt_module_end   = END_MODULE + TERM

allowed = stmt_module_begin | stmt_module_end | BLANK_LINE 

expr = pyp.OneOrMore(allowed)

def tryTopParseString(expr,t):
    try:
        print(expr.parseString(t))
    except Exception as e:
        raise e

testdata = []
testdata.append("""module test
end module test
""")
testdata.append("""module test; end module test
""")
testdata.append("""module test

end module test
""")

for t in testdata:
    try:
        print(expr.parseString(t))
    except Exception as e:
        raise e