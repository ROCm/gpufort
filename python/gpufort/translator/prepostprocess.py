# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import re

from . import opts


def preprocess_fortran_statement(statement):
    """Performs the following operations:
    - replace power (**) expression by func call expression
    # TODO handle directly via arithmetic expression grammar
    """
    result = statement
    if "**" in result:
        result = power.transformString(result)
    return result


def prepare_fortran_snippet(fortran_snippet):
    """Converts the Fortran snippet to lower case as
    we convert into a case-sensitive C language (preserves case in comments).
    Furthermore, applies a number of hacks that were necessary to get the parser work.
    """
    result = pIgnore.sub("", fortran_snippet)
    result = p_else_if.sub("else if", result)
    result = power.transformString(result)
    return result


def postprocess_c_snippet(c_snippet):
    to_hip = opts.gpufort_cpp_symbols
    for key, subst in to_hip.items():
        c_snippet = re.sub(r"\b" + key + r"\b", subst, c_snippet)
    return c_snippet
