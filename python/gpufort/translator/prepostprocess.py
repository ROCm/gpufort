# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import re

from gpufort import util

from . import opts
from . import tree

def preprocess_fortran_statement(statement):
    """Performs the following operations:
    - insert whitespaces between identifiers and operators
      with leading and trailing edge.
    - replace power (**) expression by func call expression
    # TODO handle directly via arithmetic expression grammar
    """
    result = ""
    for tk in util.parsing.tokenize(statement):
        add_ws = (tk[1:-1].isidentifier()
                 and tk[0] == "."
                 and tk[-1] == ".")
        add_ws = (add_ws 
                 or (tk.lower() not in ["e","d"]
                 and tk.isidentifier()))
        if add_ws:
            result += "".join([" ",tk," "])
        else:
            result += tk
    if "**" in result:
        criterion = True
        while criterion:
            old_result = result
            result = tree.grammar.power.transformString(result)
            criterion = old_result != result
    return result

def postprocess_c_snippet(c_snippet):
    to_hip = opts.gpufort_cpp_symbols
    for key, subst in to_hip.items():
        c_snippet = re.sub(r"\b" + key + r"\b", subst, c_snippet)
    return c_snippet
