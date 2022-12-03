# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import textwrap

from gpufort import util

from .. import opts

import pyparsing

class TTNode(object):
    
    def compare_label(self,label1,label2):
        """:return: If this bel equals argument, case is ignored.
        """
        #todo: introduce base class for statements, not all nodes can be labelled
        if label2 == None and label1 == None:
            return True
        if label2 != None and label1 != None:
            return label2.lower() == label1.lower()
        else:
            return False 

    def __init__(self,tokens=[]):
        self._init()
        self._assign_fields(tokens)

    def _init(self):
        self.parent = None
        self.numeric_label = None
    
    def _assign_fields(self, tokens):
        pass

    def __str__(self):
        return self.__class__.__name__ + ':' + str(self.__dict__)

    def child_nodes(self):
        """yield from empty tuple"""
        yield from ()
    
    def walk_preorder(self):
        yield self
        for child in self.child_nodes():
            print(str(type(child))+" of "+str(type(self)))
            yield from child.walk_preorder()
    
    def walk_postorder(self):
        for child in self.child_nodes():
            yield from child.walk_postorder()
        yield self

    def fstr(self):
        """:return: Fortran representation."""
        assert False, "Must be implemented by subclass"

    def cstr(self):
        """:return: C/C++ representation."""
        assert False, "Must be implemented by subclass"

    #__repr__ = __str__

class TTDummy(TTNode):
    """A node with user-defined C/C++ and Fortran
    representations.
    """

    def __init__(self,cstr,fstr):
        """:param str cstr: The C/C++ representation.
        :param str fstr: The Fortran representation."""
        self._cstr = cstr
        self._fstr = fstr 
    def cstr(self):
        return self._cstr
    def fstr(self):
        return self._fstr

class TTNone(TTNode):
    """Node modelling non-existing nodes or empty lists."""
    def __init__(self, tokens=[]):
        self.parent = None
        pass
    def __len__(self):
        return 0
    def __iter__(self):
        return None
    def child_nodes(self):
        return []
    def cstr(self):
        return "" 
    def fstr(self):
        return ""

class TTStatement(TTNode):
    pass

class TTContainer(TTStatement):
    """Container node for manual parser construction.
    """
    def __init__(self, tokens=[]):
        self._init(opts.single_level_indent)
        self._assign_fields(tokens)
   
    def _init(self,indent=""):
        TTNode._init(self)
        self.body = []
        self.named_label = None
        self.indent = indent 

    def __len__(self):
        return len(self.body)

    def __iter__(self):
        return iter(self.body)
 
    def append(self, node):
        self.body.append(node)

    def child_nodes(self):
        yield from self.body

    def header_cstr(self):
        return ""
    
    def footer_cstr(self):
        return ""

    def body_cstr(self):
        result = [make_cstr(child).rstrip() for child in self.body]
        return textwrap.indent("\n".join(result),self.indent)

    def cstr(self):
        body_content = self.cstr(self)
        return "{}{}\n{}".format(\
            self.header_cstr(),
            body_content,
            self.footer_cstr())

class TTRoot(TTContainer):

    def __init__(self):
        TTContainer._init(self)
