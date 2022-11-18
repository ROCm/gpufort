# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#from translator_base import *
import pyparsing
import ast

from gpufort import util

from .. import opts
from .. import conv
from .. import prepostprocess

from . import base
from . import traversals

import enum

class TTSimpleToken(base.TTNode):

    def _assign_fields(self, tokens):
        self._text = " ".join(tokens)

    def cstr(self):
        return "{};".format(self._text.lower())

    def fstr(self):
        return str(self._text)

class TTReturn(base.TTNode,base.FlowStatementMarker):

    def _assign_fields(self, tokens):
        self._result_name = ""

    def cstr(self):
        if self._result_name != None and len(self._result_name):
            return "return " + self._result_name + ";"
        else:
            return "return;"

    def fstr(self):
        return "return"

class TTLabel(base.TTNode,base.FlowStatementMarker):

    def _assign_fields(self, tokens):
        self._label = tokens[0]
        self.is_exit_marker = False
        self.is_standalone = False

    def cstr(self):
        result = "_"+self._label+":"
        if self.is_exit_marker:
            result = "_" + result
        if self.is_standalone:
            result += " ;"
        return result

class TTContinue(base.TTNode,base.FlowStatementMarker):
    def cstr(self):
        return ";"

    def fstr(self):
        return "continue"

class TTCycle(base.TTNode,base.FlowStatementMarker):

    def _assign_fields(self, tokens):
        self._result_name = ""
        self._in_loop = True

    def cstr(self):
        if self._label != None:
            # cycle label in loop is prefixed by single "_"
            return "goto _{};".format(self._label)    
        else:
            if self._in_loop:
                return "continue;"
            elif self._result_name != None and len(self._result_name):
                return "return " + self._result_name + ";"
            else:
                return "return;"

    def fstr(self):
        return "cycle {}".format(self.label)

class TTExit(base.TTNode,base.FlowStatementMarker):

    def _assign_fields(self, tokens):
        self._label = tokens[0]
        self._result_name = ""
        self._in_loop = True

    def cstr(self):
        if self._label != None:
            # exit label after loop is prefixed by "__"
            return "goto __{};".format(self._label)    
        else:
            if self._in_loop:
                return "break;"
            elif self._result_name != None and len(self._result_name):
                return "return " + self._result_name + ";"
            else:
                return "return;"

    def fstr(self):
        return "exit {}".format(self.label)

class TTUnconditionalGoTo(base.TTNode,base.FlowStatementMarker):
    #todo: More complex expressions possible with jump label list
    #todo: Target numeric label can be renamed to identifier (deleted Fortran feature)
    def _assign_fields(self, tokens):
        self._label = tokens[0]

    def cstr(self):
        return "goto _{};".format(self._label.rstrip("\n"))

class TTBlank(base.TTNode):
    
    def _assign_fields(self, tokens):
        self._text = tokens[0] 

    def cstr(self):
        return self._text 

    def fstr(self):
        return self._text 

class TTCommentedOut(base.TTNode):

    def _assign_fields(self, tokens):
        self._text = " ".join(tokens)

    def cstr(self):
        return "// {}".format(self._text)

class TTIgnore(base.TTNode):

    def cstr(self):
        return ""

class TTSubroutineCall(base.TTNode):

    def _assign_fields(self, tokens):
        self._subroutine = tokens[0]

    def cstr(self):
        return self._subroutine.cstr() + ";"
 
class TTAssignment(base.TTNode):

    def _assign_fields(self, tokens):
        self.lhs, self.rhs = tokens
        self._type = None
        self._rank = None
        self._bytes_per_element = None   
 
    @property
    def type(self):
        assert self._type != None
        return self._type
    @type.setter
    def type(self,typ):
        self._type = typ
    @property
    def rank(self):
        assert self._rank != None
        return self._rank
    @rank.setter
    def rank(self,rank):
        self._rank = rank
    @property
    def bytes_per_element(self):
        assert self._bytes_per_element != None
        return self._bytes_per_element
    @bytes_per_element.setter
    def bytes_per_element(self,bytes_per_element):
        self._bytes_per_element = bytes_per_element

    def child_nodes(self):
        yield self._lhs; yield self._rhs
    def cstr(self):
        return self._lhs.cstr() + "=" + self._rhs.cstr() + ";\n"
    def fstr(self):
        return self._lhs.fstr() + "=" + self._rhs.fstr() + ";\n"

class TTComplexAssignment(base.TTNode):

    def _assign_fields(self, tokens):
        self.lhs, self.rhs = tokens
        self._type = None
        self._rank = None
        self._bytes_per_element = None   
 
    @property
    def type(self):
        assert self._type != None
        return self._type
    @type.setter
    def type(self,typ):
        self._type = typ
    @property
    def rank(self):
        assert self._rank != None
        return self._rank
    @rank.setter
    def rank(self,rank):
        self._rank = rank
    @property
    def bytes_per_element(self):
        assert self._bytes_per_element != None
        return self._bytes_per_element
    @bytes_per_element.setter
    def bytes_per_element(self,bytes_per_element):
        self._bytes_per_element = bytes_per_element

    def child_nodes(self):
        yield self.lhs; yield self.rhs
    def cstr(self):
        """Expand the complex assignment.
        """
        result = ""
        result += "{}.x = {};\n".format(traversals.make_cstr(self.lhs),
                                        traversals.make_cstr(self.rhs._real))
        result += "{}.y = {};\n".format(traversals.make_cstr(self.lhs),
                                        traversals.make_cstr(self.rhs._imag))
        return result

class TTMatrixAssignment(base.TTNode):

    def _assign_fields(self, tokens):
        self.lhs, self.rhs = tokens
        self._type = None
        self._rank = None
        self._bytes_per_element = None   
 
    @property
    def type(self):
        assert self._type != None
        return self._type
    @type.setter
    def type(self,typ):
        self._type = typ
    @property
    def rank(self):
        assert self._rank != None
        return self._rank
    @rank.setter
    def rank(self,rank):
        self._rank = rank
    @property
    def bytes_per_element(self):
        assert self._bytes_per_element != None
        return self._bytes_per_element
    @bytes_per_element.setter
    def bytes_per_element(self,bytes_per_element):
        self._bytes_per_element = bytes_per_element

    def child_nodes(self):
        yield self.lhs; yield self.rhs
    def cstr(self):
        """
        Expand the matrix assignment.
        User still has to fix the ranges manually. 
        """
        result = "// TODO: fix ranges"
        for expression in self.rhs:
            result += traversals.make_cstr(
                self.lhs) + argument + "=" + flatten_arith_expr(
                    expression) + ";\n"
        return result

class TTDo(base.TTContainer):

    def _assign_fields(self, tokens):
        self._begin, self._end, self._step, self.body = tokens
        self.numeric_do_label = None
    @property
    def index(self):
        return self._begin._lhs
    @property
    def first(self):
        return self._begin._rhs
    @property
    def last(self):
        return self._end
    @property
    def step(self):
        return self._step
    def has_step(self):
        return self._step != None
    def child_nodes(self):
        yield self.body; yield self._begin; yield self._end; yield self._step

class TTUnconditionalDo(base.TTContainer):
    def _assign_fields(self, tokens):
        self.body = tokens[0]
        self.numeric_do_label = None
    def child_nodes(self):
        return []
    def header_cstr(self):
        return "while (true) {{\n"
    def footer_cstr(self):
        return "}\n" 

class TTBlock(base.TTContainer):
    def _assign_fields(self, tokens):
        self.indent = "" # container of if/elseif/else branches, so no indent
    def header_cstr(self):
        return "{{\n"
    def footer_cstr(self):
        return "}\n" 

class TTIfElseBlock(base.TTContainer):
    def _assign_fields(self, tokens):
        self.indent = "" # container of if/elseif/else branches, so no indent

class TTIfElseIf(base.TTContainer):

    def _assign_fields(self, tokens):
        self._else, self._condition, self.body = tokens

    def child_nodes(self):
        yield self._condition; yield self.body
    
    def header_cstr(self):
        prefix = self._else+" " if self._else != None else ""
        return "{}if ({}) {{\n".format(prefix,traversals.make_cstr(self._condition))
    def footer_cstr(self):
        return "}\n" 

class TTElse(base.TTContainer):
    
    def header_cstr(self):
        return "else {\n"
    def footer_cstr(self):
        return "}\n" 

    def cstr(self):
        body_content = base.TTContainer.cstr(self)
        return "{}{}\n{}".format(
            self.header_cstr(),
            body_content,
            self.footer_cstr())

class TTSelectCase(base.TTContainer):
    def _assign_fields(self, tokens):
        self.selector = tokens[0]
        self.indent = "" # container of if/elseif/else branches, so no indent
    def header_cstr(self):
        return "switch ({}) {{\n".format(self.selector)
    def footer_cstr(self):
        return "}\n" 

class TTCase(base.TTContainer):

    def _assign_fields(self, tokens):
        self.cases, self.body = tokens

    def child_nodes(self):
        yield self.cases; yield self.body
    
    def header_cstr(self):
        result = ""
        for case in self.cases:
            result += "case ("+traversals.make_cstr(case)+"):\n"
        return result
    def footer_cstr(self):
        return "  break;\n" 

class TTCaseDefault(base.TTContainer):

    def _assign_fields(self, tokens):
        self.body = tokens[0]

    def child_nodes(self):
        yield self.body
    
    def header_cstr(self):
        return "default:\n"
    def footer_cstr(self):
        return "  break;\n" 

class TTDoWhile(base.TTContainer):

    def _assign_fields(self, tokens):
        self._condition, self.body = tokens
        self.numeric_do_label = None

    def child_nodes(self):
        yield self._condition; yield self.body
    
    def header_cstr(self):
        return "while ({0}) {{\n".format(
          traversals.make_cstr(self._condition)
        )
    def footer_cstr(self):
        return "  break;\n" 

def set_fortran_parse_actions(grammar):
    grammar.assignment.setParseAction(TTAssignment)
    grammar.matrix_assignment.setParseAction(TTMatrixAssignment)
    grammar.complex_assignment.setParseAction(TTComplexAssignment)
    grammar.fortran_subroutine_call.setParseAction(TTSubroutineCall)
