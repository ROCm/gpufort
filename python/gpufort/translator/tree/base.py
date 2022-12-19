# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import textwrap

from gpufort import util

from .. import opts

import pyparsing

class TTNode(object):
    # TODO move labeling to TTStatement    

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
        """Pre-order tree walk iterator, i.e. yields
        the current node before its children.
        """
        yield self
        for child in self.child_nodes():
            yield from child.walk_preorder()
    
    def walk_postorder(self):
        """Post-order tree walk iterator, i.e. yields
        the current node after its children.
        """
        for child in self.child_nodes():
            yield from child.walk_postorder()
        yield self

    def enter_and_leave(self):
        """Iterator that yields a tuple consisting of the current node 
        and a flag if the node was entered (False) or left (True).
        The flag thus indicates if the iterator is going up the tree again with respect
        to that node.
        """
        yield (self,False) # yield at enter
        for child in self.child_nodes():
            yield from child.enter_and_leave()
        yield (self,True) # yield at leave

    def fstr(self):
        """:return: Fortran representation."""
        assert False, "Must be implemented by subclass"

    def cstr(self):
        """:return: C/C++ representation."""
        assert False, "Must be implemented by subclass"

    #__repr__ = __str__

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

    def child_statements(self):
        yield from ()

    def walk_statements_preorder(self):
        """Pre-order tree walk iterator, i.e. yields
        the current statement before its children.
        """
        yield self
        for child in self.child_statements():
            yield from child.walk_statements_preorder()
    
    def walk_statements_postorder(self):
        """Post-order tree walk iterator, i.e. yields
        the current statement after its children.
        """
        for child in self.child_statements():
            yield from child.walk_statements_postorder()
        yield self

    def enter_and_leave_statements(self):
        """Iterator that yields a tuple consisting of the current statement 
        and a flag if the statement was entered (False) or left (True).
        The flag thus indicates if the iterator is going up the tree again with respect
        to that statement.
        """
        yield (self,False) # yield at enter
        for child in self.child_statements():
            yield from child.enter_and_leave_statements()
        yield (self,True) # yield at leave

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

    def child_statements(self):
        yield from self.body

    def __len__(self):
        return len(self.body)

    def __iter__(self):
        return iter(self.body)
 
    def append(self, node):
        self.body.append(node)

    def child_nodes(self):
        yield from self.body

    def header_cstr(self):
        """C representation of the header of this container.
        :note: Intended to be overriden by subclass.
        :note: Implementation may not open new scope, add indentation,
               or add line break. This should be done by the render routine.
        """
        return ""
    
    def footer_cstr(self):
        """C representation of the footer of this container.
        :note: Intended to be overriden by subclass.
        :note: Implementation may not close scope, add indentation, and add line break.
               This should be done by the render routine.
        """
        return ""

class TTRoot(TTContainer):

    def __init__(self):
        TTContainer._init(self)

def to_cstr(expr):
    """:return: a C++ representation if the 
    expr is an instance of TTNode besides TTContainer.
    Returns `expr` itself if it is string."""
    if type(expr) == str:
        return expr
    elif isinstance(expr,TTContainer):
        assert False, "no support for container statements"
    elif isinstance(expr,TTNode):
        return expr.cstr()
    assert False, "no support for type '{}'".format(expr)
