# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from . import opts


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


def convert_to_c_type(f_type, kind, default=None):
    """:return: An equivalent C datatype for a given Fortran type, e.g. `double` for a `REAL*8`.
    :param f_type: The original Fortran type, e.g. `REAL` for a `REAL*8`.
    :param kind: The kind of the Fortran type, e.g. `8` for a `REAL*8`.
    :rtype: str
    """
    assert type(f_type) is str
    if kind is None:
        kind = ""
    assert type(kind) is str, "{}, {}".format(kind, type(kind))
    kind_lower = kind.lower()
    return opts.fortran_2_c_type_map.get(f_type.lower(), {
        kind_lower: default
    }).get(kind_lower, "UNKNOWN")
