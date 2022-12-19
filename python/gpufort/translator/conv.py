# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from . import opts

def bytes_per_element(ftype,kind):
    """Lookup number of bytes required to store one instance of the type. 
    :note: As Fortran allows user-defined kinds, we use
           bytes per element to uniquely determine a datatype.
    :note: argument 'kind' might be not be lower case if the property 
           was derived from a literal such as '1.0_DP'.
    :note: We assume the Fortran type ('ftype') is already lower
           case as it stems from the symbol table.
    """
    kind = kind.lower() if kind != None else None
    try:
        return opts.fortran_type_2_bytes_map[ftype][kind]
    except KeyError as ke:
        if ftype == "type":
            return None
        else:
            raise util.error.LookupError("could not find bytes per element for"
              + " type '{}' and kind '{}' in translator.opts.fortran_type_2_bytes_map".format(
                ftype,kind
              )
            ) from ke

def c_type(ftype,bytes_per_element):
    try:
        return opts.bytes_2_c_type[ftype][bytes_per_element]
    except KeyError as ke:
        raise util.error.LookupError((
          "could not find C type for element for"
          + " type '{}' and bytes per element '{}'"
          + " in translator.opts.fortran_type_2_bytes_map"
          ).format(ftype,bytes_per_element)
        ) from ke

# todo: deprecated
def num_bytes(f_type, kind, default=None):
    """:return: number of bytes to store datatype 'f_type' of kind 'kind'. Expression might contain parameters."""
    assert type(f_type) is str
    assert kind == None or type(kind) is str
    if kind == None:
        kind_lower = ""
    else:
        kind_lower = kind.lower()
    f_type_lower = f_type.lower().replace(" ", "")
    if f_type_lower in opts.fortran_type_2_bytes_map and\
       kind_lower in opts.fortran_type_2_bytes_map[f_type_lower]:
        return opts.fortran_type_2_bytes_map[f_type_lower][kind_lower]
    elif f_type_lower == "complex":
        return "2*(" + kind_lower + ")"
    else:
        return "(" + kind_lower + ")"


# todo: deprecated
def convert_to_c_type(f_type, kind, default=None):
    """:return: An equivalent C datatype for a given Fortran type, e.g. `double` for a `REAL*8`.
    :param f_type: The original Fortran type, e.g. `REAL` for a `REAL*8`.
    :param kind: The kind of the Fortran type, e.g. `8` for a `REAL*8`.
    :rtype: str
    :raise ValueError: if the Fortran type could not be transformed into a C type.
    """
    assert type(f_type) is str
    if kind is None:
        kind = ""
    assert type(kind) is str, "{}, {}".format(kind, type(kind))
    kind_lower = kind.lower()
    result = opts.fortran_2_c_type_map.get(f_type.lower(), {
        kind_lower: default
    }).get(kind_lower, None)
    if result == None:
        raise ValueError("Fortran type '{}' of kind '{}' could not be translated to C type.".format(\
            str(f_type),str(kind)))
    return result

# todo: deprecated
def get_operator_name(op):
    if op[0] == ".":
        return op.replace(".","")
    elif op == "+":
        return "add"
    elif op == "*":
        return "mult"
    else:
        return op
    
def reduction_c_init_val(fortran_op,c_type):
    """:return: Initial value for the given Fortran reduction op
    and the given C type."""
    fortran_op = fortran_op.lower()
    if fortran_op == "max":
        return "-std::numeric_limits<{}>::max()".format(c_type)
    elif fortran_op == "min":
        return "+std::numeric_limits<{}>::max()".format(c_type)
    elif:
        return "-1" # must be integer, no unsigned int in Fortran
    else:
        mappings = {
          "+": "0",
          "*": "1",
          "ior": "0",
          "ieor": "0",
          ".and.": "true",
          ".or.": "false",
          ".eqv.": "true",
          ".neqv.": "false",
        }
            return mappings[fortran_op]
   
def reduction_c_op(fortran_op):
    mappings = {
     "+": "+",
     "*": "*",
     "max": "max",
     "min": "min",
     "iand": "&",
     "ior": "|",
     "ieor": "^",
     ".and.": "&&",
     ".or.": "||",
     ".eqv.": "==",
     ".neqv.": "!=",
    }
    return mappings[fortran_op.lower()]
