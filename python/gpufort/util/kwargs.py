# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
def get_value(key, default, **kwargs):
    """:return: Tuple consisting of the value for the key or the default and
                a flag that indicates if the keyword argument has been found."""
    for k, v in kwargs.items():
        if key == k:
            return (v,True) 
    return (default,False)


def set_from_kwargs(obj, attr, default, **kwargs):
    """Adds an attribute to 'obj' and sets its
    value to that of keyword argument if found.
    Otherwise, sets it to the default value.
    :return: If the keyword argument has been found."""
    value, found = get_value(attr, default, **kwargs)
    setattr(obj, attr, value)
    return found
