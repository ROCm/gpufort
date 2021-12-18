# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import pyparsing as pyp

number     = pyp.pyparsing_common.number.copy().setParseAction(lambda x: str(x))
identifier = pyp.pyparsing_common.identifier
terminal   = identifier | number

expr = pyp.infixNotation(terminal, [
    (pyp.oneOf('* /'), 2, pyp.opAssoc.LEFT),
    (pyp.oneOf('+ -'), 2, pyp.opAssoc.LEFT),
    (pyp.oneOf('-'), 1, pyp.opAssoc.LEFT),
]) + pyp.stringEnd()

print(expr.parseString("-5 + 3"))