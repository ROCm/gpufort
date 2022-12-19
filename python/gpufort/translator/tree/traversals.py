# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import pyparsing

from . import base

def no_action(*args,**kwargs):
    pass

def no_crit(*args,**kwargs):
    return False

def traverse(expr,
             enter_action,     # type: Callable[expr,parents,*args,**kwargs]
             leave_action,     # type: Callable[expr,parents,*args,**kwargs]
             termination_crit, # type: Callable[expr,parents,*args,**kwargs] -> bool
             *args,
             **kwargs):
    parents = []
    def descend(current,parents):
        nonlocal args
        nonlocal kwargs
        if termination_crit(current,parents,*args,**kwargs):
            return
        enter_action(current,parents,*args,**kwargs)
        parents.append(expr)
        if isinstance(current,(pyparsing.ParseResults,list)):
            # todo: still need to check for list?
            for el in current:
                descend(el,parents)
        elif isinstance(current, base.TTNode):
            for child in current.child_nodes():
                descend(child,parents)
        parents.pop(-1)
        leave_action(current,parents,*args,**kwargs)
    descend(expr,parents)

def find_all_matching(obj, filter_expr=lambda x: True, N=-1):
    """
    Find all nodes in tree of type 'searched_type'.
    """
    result = []

    def descend_(curr):
        if N > 0 and len(result) > N:
            return
        if filter_expr(curr):
            result.append(curr)
        if isinstance(curr,pyparsing.ParseResults) or\
           isinstance(curr,list):
            for el in curr:
                descend_(el)
        elif isinstance(curr, base.TTNode):
            for child in curr.child_nodes():
                descend_(child)
    descend_(obj)
    return result
def find_first_matching(obj, filter_expr=lambda x: True):
    """
    Find first node in tree where the filte returns true.
    """
    result = find_all_matching(obj, filter_expr, N=1)
    if len(result):
        return result[0]
    else:
        return None


def find_all(obj, ):
    """Find all nodes in tree of type 'searched_type'.
    """
    result = []

    def descend(curr):
        if N > 0 and len(result) > N:
            return
        if type(curr) is searched_type:
            result.append(curr)
        if isinstance(curr,(pyparsing.ParseResults,list)):
            for el in curr:
                descend(el)
        elif isinstance(curr, base.TTNode):
            for child in curr.child_nodes():
                descend(child)
    descend(obj)
    return result
def find_all(obj, searched_type, N=-1):
    """Find all nodes in tree of type 'searched_type'.
    """
    result = []

    def descend(curr):
        if N > 0 and len(result) > N:
            return
        if type(curr) is searched_type:
            result.append(curr)
        if isinstance(curr,(pyparsing.ParseResults,list)):
            for el in curr:
                descend(el)
        elif isinstance(curr, base.TTNode):
            for child in curr.child_nodes():
                descend(child)
    descend(obj)
    return result
def find_first(obj, searched_type):
    """Find first node in tree of type 'searched_type'.
    """
    result = find_all(obj, searched_type, N=1)
    if len(result):
        return result[0]
    else:
        return None

def make_cstr(obj):
    if obj is None:
        return ""
    try:
        return obj.cstr().lower()
    except Exception as e:
        if isinstance(obj,pyparsing.ParseResults) or\
           isinstance(obj,list):
            result = ""
            for child in obj:
                result += make_cstr(child)
            return result
        elif type(obj) in [bool, int, float, str]:
            return str(obj).lower()
        else:
            raise Exception("cannot convert object of type '{}' to C expression".format(type(obj))) from e


def make_fstr(obj):
    if obj is None:
        return ""
    try:
        return obj.fstr()
    except Exception as e:
        if isinstance(obj,pyparsing.ParseResults) or\
           isinstance(obj,list):
            result = ""
            for child in obj:
                result += make_fstr(child)
            return result
        elif type(obj) in [int, float, str]:
            return str(obj)
        else:
            raise Exception("cannot convert object of type '{}' to Fortran expression".format(type(obj))) from e
