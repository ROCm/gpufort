# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import textwrap

from gpufort import util

from .. import opts
from . import grammar

import pyparsing

# Ternary state
False3, Unknown3, True3 = -1, 0, 1

class TTNode(object):

    def __init__(self, s, loc, tokens):
        self._input = s
        self._location = loc
        self.parent = None
        self._assign_fields(tokens)

    def __str__(self):
        return self.__class__.__name__ + ':' + str(self.__dict__)

    def _assign_fields(self, tokens):
        pass

    def children(self):
        result = []
        for key, value in self.__dict__.items():
            if key != "parent":
                result.append(value)
        return result

    def c_str(self):
        pass


    #__repr__ = __str__
class TTContainer(TTNode):
    """Container node for manual parser construction.
    """
    def __init__(self, s="", loc=0, tokens=[]):
        self._input = s
        self._location = loc
        self.parent = None
        self.body   = []
        self.indent = opts.indent 
        self._assign_fields(tokens)
    
    def append(self, node):
        self.body.append(node)

    def children(self):
        return self.body

    def c_str(self):
        result = [make_c_str(child).rstrip() for child in self.body]
        return textwrap.indent("\n".join(result),self.indent)

class TTRoot(TTContainer):
    pass


def f_keyword(keyword):
    if opts.keyword_case == "upper":
        return keyword.upper()
    elif opts.keyword_case == "lower":
        return keyword.lower()
    elif opts.keyword_case == "camel":
        return keyword[0].upper() + keyword[1:].lower()
    else:
        return keyword


ADDITIONS = "cuf kernel kind amax1 amin1" # TODO
KEYWORDS = " ".join([
    grammar.CUDA_FORTRAN_KEYWORDS_STR, grammar.FORTRAN_INTRINSICS_STR,
    grammar.F08_KEYWORDS_STR, ADDITIONS
])
KEYWORDS = KEYWORDS.lower().split(" ")


def find_all_matching(body, filter_expr=lambda x: True, N=-1):
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
        elif isinstance(curr, TTNode):
            for child in curr.children():
                descend_(child)

    descend_(body)
    return result


def find_first_matching(body, filter_expr=lambda x: True):
    """
    Find first node in tree where the filte returns true.
    """
    result = find_all_matching(body, filter_expr, N=1)
    if len(result):
        return result[0]
    else:
        return None


def find_all(body, searched_type, N=-1):
    """Find all nodes in tree of type 'searched_type'.
    """
    result = []

    def descend(curr):
        if N > 0 and len(result) > N:
            return
        if type(curr) is searched_type:
            result.append(curr)
        if isinstance(curr,pyparsing.ParseResults) or\
           isinstance(curr,list):
            for el in curr:
                descend(el)
        elif isinstance(curr, TTNode):
            for child in curr.children():
                descend(child)

    descend(body)
    return result


def find_first(body, searched_type):
    """Find first node in tree of type 'searched_type'.
    """
    result = find_all(body, searched_type, N=1)
    if len(result):
        return result[0]
    else:
        return None


def make_c_str(obj):
    if obj is None:
        return ""
    try:
        return obj.c_str().lower()
    except Exception as e:
        if isinstance(obj,pyparsing.ParseResults) or\
           isinstance(obj,list):
            result = ""
            for child in obj:
                result += make_c_str(child)
            return result
        elif type(obj) in [bool, int, float, str]:
            return str(obj).lower()
        else:
            raise e


def make_f_str(obj):
    if obj is None:
        return ""
    try:
        return obj.f_str()
    except Exception as e:
        if isinstance(obj,pyparsing.ParseResults) or\
           isinstance(obj,list):
            result = ""
            for child in obj:
                result += make_f_str(child)
            return result
        elif type(obj) in [int, float, str]:
            return str(obj)
        else:
            raise e


def flatten_body(body):
    #TODO: Check if this is the same as make_c_str()
    def descend(statement):
        term = ""
        if isinstance(statement,pyparsing.ParseResults) or\
           isinstance(statement,list):
            for el in statement:
                term += descend(el)
            return term
        elif isinstance(statement, TTNode):
            try:
                return statement.c_str()
            except Exception as e:
                raise e
        else:
            return ""

    return descend(body)
