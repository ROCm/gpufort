# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import pyparsing as pyp

number     = pyp.pyparsing_common.number.copy().setParseAction(lambda x: str(x))
identifier = pyp.pyparsing_common.identifier
terminal   = identifier | number

expr = pyp.infixNotation(terminal, [
    (pyp.Regex('[\-+]'), 1, pyp.opAssoc.RIGHT),
    (pyp.Regex('[*/+\-]'), 2, pyp.opAssoc.LEFT),
]) + pyp.stringEnd()

print(expr.parseString("-5 + 3"))
print(expr.parseString("-5"))
print(expr.parseString("-( a + (b-1))"))
