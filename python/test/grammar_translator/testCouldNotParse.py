#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
from pyparsing import *

identifier = pyparsing_common.identifier
EQ = Literal("=")
PM = Regex("[+-]")
EOL = LineEnd().suppress()

assignment = identifier + EQ + infixNotation(identifier, [ (PM, 2, opAssoc.LEFT) ], )
notAssignment = ~assignment

#grammar = assignment | unknownLine
#grammar = OneOrMore(unknownLine)

input="""
a = b + c,
a = b + 2,
a = b + d,
a = b * 2, 
"""

result = notAssignment.searchString(input)
print(result.asList())

