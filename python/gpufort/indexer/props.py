# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.

def has_any_attribute(irecord,attributes):
    """Checks if the index record has any of the provided
    attributes. 
    :param irecord: index record of a variable, type, procedure, program, or module.
    :param list(string) attributes: List of attributes
    """
    for attrib in attributes:
        if attrib in irecord["attributes"]:
            return True
    return False

def index_var_is_array(ivar):
    """If the index var is an array."""
    return ivar["rank"] > 0

def index_var_is_assumed_size_array(ivar):
    """If the index var is an assumed-size array,
    which indicated by an asterisk in the array's
    last dimension's bounds specification.
    """
    if index_var_is_array(ivar):
        return "*" in ivar["bounds"][-1]
    else:
        return False

def index_var_known_rank(ivar):
    """Returns the minimum rank that the array has, i.e.
    the number of dimensions that have known upper
    and lower bounds. In case of assumed-size arrays,
    the last dimension's upper bound is not known, i.e.
    the 'known rank' is reduced by 1. 
    """
    if index_var_is_assumed_size_array(ivar):
        return ivar["rank"] - 1
    else:
        return ivar["rank"]
