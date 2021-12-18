#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
from pyparsing import *

operand = Combine(Optional(Literal("-"),default="") + pyparsing_common.identifier) | pyparsing_common.number
expr = pyparsing_common.identifier + Literal("=") + infixNotation(operand,
        [
        (oneOf('* /'), 2, opAssoc.LEFT),
        (oneOf('+ -'), 2, opAssoc.LEFT),
        ])
print(expr.parseString("a = -a + (-5)"))