# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.

"""Tree nodes that are introduced in the context of tree transformations.
"""

from . import base

class TTDummy(base.TTNode):
    """A node with user-defined C/C++ and Fortran
    representations, specified as string.
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


class TTDummyStatement(base.TTStatement):
    """A node with user-defined C/C++ and Fortran
    representations, specified as string.
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

class TTUnrolledArrayAssignment(base.TTStatement):
    """Specific assignment class for unrolled array assignments."""

    def _assign_fields(self, tokens):
        self.lhs, self.rhs = tokens
 
    @property
    def type(self):
        return self.lhs.type
    @property
    def rank(self):
        return 0 
    @property
    def bytes_per_element(self):
        return self.lhs.bytes_per_element
    
    def walk_values_ignore_args(self):
        yield from self.lhs.walk_values_ignore_args()
        yield from self.rhs.walk_values_ignore_args()
    
    def child_nodes(self):
        yield self.lhs
        yield self.rhs
    def cstr(self):
        return self.lhs.cstr() + "=" + self.rhs.cstr() + ";\n"
    def fstr(self):
        assert False, "no Fortran representation"

class TTSubstitution(base.TTStatement):
    """A statement with original and substituted tree. 
    """

    def __init__(self,orig,subst):
        """Constructor.
        :param str orig: The original node.
        :param str subst: The substitute node.
        """
        self.orig = orig
        self.subst = subst 

    def child_nodes(self):
        yield from self.subst.child_nodes()

    def fstr(self):
        self.orig.fstr()
    
    def cstr(self):
        return self.subst.cstr()
