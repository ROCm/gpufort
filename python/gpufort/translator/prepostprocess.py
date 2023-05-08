
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
            result += " " + tk + " "
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

def modernize_fortran_function_name(name):
    """Renames archaic Fortran intrinsic names
    to their modern variant."""
    name_lower = name.lower()
    # conversion routines:
    # achar
    # aint
    # anint
    # aimag
    # special case:
    # algama -> lgamma
    # remove a:
    # archaic math functions
    if name_lower in [
    "alog",
    "alog10",
    "amod",
    ]:
      return name_lower[1:]
    # archaic min, max variants:
    elif name_lower in [
    "amax0",
    "amax1",
    ]:
      return name_lower[1:-1]
    elif name_lower in [
    "amin0",
    "amin1",
    ]:
      return name_lower[1:-1]
    # inverse trignometric functions:
    #asin
    #acos
    #acosd
    #acosh
    #asind
    #asinh
    #atan
    #atan2
    #atan2d
    #atand
    #atanh
    # conversion routines
    #dble
    #dcmplx
    #dconjg
    #dimag
    #dreal
    #dint
    #dfloat
    # other:
    #dshiftl
    #dshiftr
    #dtime
    elif name_lower in [
    "dabs",
    "dacos",
    "dacosd",
    "dacosh",
    "dasin",
    "dasind",
    "dasinh",
    "datan",
    "datan2",
    "datan2d",
    "datand",
    "datanh",
    "dbesj0",
    "dbesj1",
    "dbesjn",
    "dbesy0",
    "dbesy1",
    "dbesyn",
    "dcos",
    "dcosd",
    "dcosh",
    "dcotan",
    "dcotand",
    "ddim",
    "derf",
    "derfc",
    "dexp",
    "dgamma",
    "dlgama",
    "dlog",
    "dlog10",
    "dmax1",
    "dmin1",
    "dmod",
    "dnint",
    "dprod",
    "dsign",
    "dsin",
    "dsind",
    "dsinh",
    "dsqrt",
    "dtan",
    "dtand",
    "dtanh",
    ]:
        return name_lower[1:]
    else:
        return name_lower
