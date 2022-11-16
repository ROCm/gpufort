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

# literal expressions

class Literal(base.TTNode):
    LOGICAL = "logical"
    CHARACTER = "character"
    INTEGER = "integer"
    REAL = "real"
            
    def _assign_fields(self, tokens):
        """
        Expected inputs for 
        """
        self._raw_value = tokens[0]
        self._value = self._raw_value
        self._kind = None
        if "_" in self._raw_value:
            parts = self._raw_value.rsplit("_",maxsplit=1)
            self._value = parts[0]
            self._kind = parts[1] 
        self._bytes_per_element = None
    
    @property
    def value(self):
        return self._value
    
    @property
    def kind(self):
        return self._kind

    @property
    def bytes_per_element(self):
        assert self._bytes_per_element != None
        return self._bytes_per_element

    @bytes_per_element.setter
    def bytes_per_element(self,bytes_per_element):
        self._bytes_per_element = bytes_per_element 

    @property
    def resolved(self):
        return self._bytes_per_element != None
   
    @property
    def rank(self):
        return 0
    
    def fstr(self):
        return self._raw_value
 
class TTCharacter(Literal):
    
    def _assign_fields(self, tokens):
        self._raw_value = tokens[0]
        self._value = self._raw_value
        self._kind = None
        self._bytes_per_element = None

    @property
    def type(self):
        return Literal.CHARACTER 
   
    @property
    def len(self):
        return len(self._raw_value)-2
 
    def cstr(self):
        return '"'+self._raw_value[1:-1]+'"'

class TTLogical(Literal):

    @property
    def type(self):
        return Literal.LOGICAL

    def cstr(self):
        if self.bytes_per_element == "1":
            return "true" if self._value.lower() == ".true." else "false"
        else:
            return "1" if self._value.lower() == ".true." else "0"

class TTNumber(Literal):

    def _assign_fields(self, tokens):
        Literal._assign_fields(self,tokens)
        if "." in self._value:
            self._type = Literal.REAL 
            if self._kind == None:
                if "d" in self._value:
                    self._kind = "c_double"
        else:
            self._type = Literal.INTEGER
    
    @property
    def type(self):
        return self._type

    def cstr(self):
        # todo: check kind parameter in new semantics part
        if self._type == Literal.REAL:
            if self.bytes_per_element == 4:
                return self._value + "f"
            elif self.bytes_per_element == 8:
                return self._value.replace("d", "e")
            else:
                raise util.error.LimitationError("only single & double precision floats supported")
        elif self._type == Literal.INTEGER:
            if self.bytes_per_element == 16:
                return self._value + "LL"
            elif self.bytes_per_element == 8:
                return self._value + "L"
            else:
                # short is typically promoted to int
                return self._value
    
    def __str__(self):
        kind = self._kind if self._kind != None else "<none>"
        return "TTNumber(val:"+str(self._value)+",kind:"+kind+")"
    __repr__ = __str__

# variable expressions

class VarExpr(base.TTNode):

    def _assign_fields(self, tokens):
        self._symbol_info = None
        self._bytes_per_element = None 
    
    @property
    def partially_resolved(self):
        return self._symbol_info != None
   
    @property
    def resolved(self):
        return ( 
          self.partially_resolved
          and self._bytes_per_element != None
        )
 
    @property
    def symbol_info(self):
        assert self.partially_resolved
        return self._symbol_info 
    
    @symbol_info.setter
    def symbol_info(self,symbol_info):
        self._symbol_info = symbol_info
   
    def _get_type_defining_record(self):
        """:return: An index symbol_info that describes the type
        of an expression. Per default it is the main symbol_info."""
        return self.symbol_info 

    @property 
    def type(self):
        return self._get_type_defining_record()["f_type"]
    
    @property 
    def kind(self):
        return self._get_type_defining_record()["kind"]

    @property
    def bytes_per_element(self):
        assert self.resolved
        return self._bytes_per_element
    
    @bytes_per_element.setter
    def bytes_per_element(self,bytes_per_element):
        self._bytes_per_element = bytes_per_element 

    @property
    def ctype(self):
        assert self.resolved
        return opts.bytes_2_c_type[
          self.type][self.bytes_per_element] 

    @property 
    def rank(self):
        assert self.partially_resolved
        return self.symbol_info["rank"]

class TTIdentifier(VarExpr):
        
    def _assign_fields(self, tokens):
        self._name = tokens[0]
        VarExpr._assign_fields(self,tokens)

    @property
    def is_array(self):
        return self.rank > 0

    @property
    def is_contiguous_array(self):
        return self.is_array
    
    @property
    def is_full_array(self):
        return self.is_array
    
    @property
    def is_scalar(self):
        return not self.is_array
  
    def fstr(self):
        return str(self._name)

    def cstr(self):
        return self.fstr().lower()

    def __str__(self):    
        return "TTIdentifier(name:"+str(self._name)+")"
    __repr__ = __str__


class TTFunctionCall(VarExpr):
 
    def _assign_fields(self, tokens):
        self._name = tokens[0]
        if len(tokens) > 1:
            self.args = tokens[1:]
        else:
            self.args = []
        VarExpr._assign_fields(self,tokens)
    
    # override
    def _get_type_defining_record(self):
        symbol_info = self.symbol_info
        if "rank" in symbol_info: # is variable
            return symbol_info
        else:
            assert symbol_info["kind"] == "function"
            result_name = symbol_info["result_name"]
            ivar = next((ivar
              for ivar in symbol_info["variables"] 
              if ivar["name"] == result_name),None)
            assert ivar != None
            return ivar

    def child_nodes(self):
        for arg in self.args:
            yield arg

    def slice_args(self):
        """Returns all range args in the order of their appeareance.
        """
        # todo: double check that this does not descend
        for arg in self.args:
            if isinstance(arg,TTSlice):
                yield arg

    def is_array_expr(self):
        """:return: if this expression represents
        an array section or array element (array section of size 1)."""
        return "rank" in self.symbol_info
   
    @property
    def is_array(self):
        """If there is a index range."""
        assert self.is_array_expr()
        for i,arg in enumerate(self.args):
            if isinstance(arg,TTSlice):
                return True
        return False
        
    @property
    def is_contiguous_array(self):
        """Checks if this array section expression
        is a contiguous section of an original array.
        
        Returns true if
        - no index range has been found at all
        - or if one or more index range have been found at 
          the begin of the index argument list
            - and all bounds and step sizes are unspecified with
              the only allowed exception being the upper bound of the last index range
        - or if one index range has been found at the
          begin of the index argument list and no other index range.
        """
        assert self.is_array_expr()
        last_slice = -1 # must be initialized with -1
                        # as then: i-last==1 at i=0
        last_slice_with_ubound = -1
        last_slice_with_lbound = -1
        for i,arg in enumerate(self.args):
            if isinstance(arg,TTSlice):
                if (
                    arg.has_stride()
                    or i - last_slice != 1
                    or i > 0 and arg.has_lbound()
                    or last_slice_with_lbound >= 0
                    or last_slice_with_ubound >= 0
                  ):
                    return False
                if arg.has_lbound():
                    last_slice_with_lbound = i
                if arg.has_ubound():
                    last_slice_with_ubound = i
                last_slice = i
        # no strides in any slice at this point
        return not (
          last_slice < 0 # is scalar
          #or last_slice_with_lbound > 1 # lower bound specified for dim other than 0
          #or last_slice_with_lbound < last_slice # other ranges found after dim with lower bound
          #or (last_slice_with_ubound >= 0 # upper bound specified for lower dim
          #   and last_slice_with_ubound < last_slice)
        )
    
    @property
    def is_scalar(self):
        """Checks if none of the array index arguments 
        are index ranges.
        """
        assert self.is_array_expr()
        for arg in self.args:
            if isinstance(arg,TTSlice):
                return False
        return True

    @property
    def is_full_array(self):
        """Checks if this array section expression
        covers all elements of an original array.
        """
        assert self.is_array_expr()
        for arg in self.args:
            if (not isinstance(arg,TTSlice)
               or arg.has_lbound()
               or arg.has_ubound()
               or arg.has_stride()):
                return False
        return True
 
    def is_function_call(self):
        return not self.is_array_expr()
    
    def is_intrinsic_call(self):
        assert self.is_function_call()
        return "intrinsic" in self.symbol_info["attributes"]
   
    def is_elemental_function_call(self):
        assert self.is_function_call()
        return "elemental" in self.symbol_info["attributes"]

    def is_converter_call(self):
        """:note: Conversions are always elemental."""
        assert self.is_elemental_call()
        return "conversion" in self.symbol_info["attributes"]
 
    @property
    def name(self):
        return self._name
 
    @property 
    def rank(self):
        if self.is_array_expr():
            return len([arg for arg in self.slice_args()])
        else: 
            return self._get_type_defining_record()["rank"] 

    def name_cstr(self):
        name = traversals.make_cstr(self._name).lower()
        if self.is_array_expr():
            return name
        elif self.is_intrinsic_call():
            # todo
            name = prepostprocess.modernize_fortran_function_name(name)
            if name in [
              "max",
              "min",
              ]:
                num_args = len(self._args)
                return "max" + str(num_args)
            else:
                return name
        else:
            return name
    
    # TODO reassess
    def __max_rank_adjusted_items(self):
        if self.rank > 0:
            assert self.max_rank <= len(self.items)
            result = self.items[0:self.max_rank]
        else:
            result = []
        return result

    def __args_cstr(self,name,is_array=False,fortran_style_array_access=True):
        args = self.__max_rank_adjusted_items()
        if len(self.args):
            if (not fortran_style_array_access and is_array):
                return "[_idx_{0}({1})]".format(name, ",".join([
                    traversals.make_cstr(s) for s in args
                ])) # Fortran identifiers cannot start with "_"
            else:
                return "({})".format(
                    ",".join([traversals.make_cstr(s) for s in args]))
        else:
            return ""

    def __args_fstr(self):
        args = self.args
        if len(args):
            return "({0})".format(",".join(
                traversals.make_fstr(el) for el in args))
        else:
            return ""

    def cstr(self):
        name = self.name_cstr()
        return "".join([
          name,
          self.__args_cstr(
            name,
            self.is_array_expr(),
            opts.fortran_style_array_access)
        ])
    
    def fstr(self):
        name = traversals.make_fstr(self._name)
        return "".join([
            name,
            self.__args_fstr()
            ])
    
    def __str__(self):
        return "TTFunctionCall(name:"+str(self._name)+",is_array_expr:"+str(self.is_array_expr())+")"
    __repr__ = __str__

class TTDerivedTypePart(base.TTNode):

    def _assign_fields(self, tokens):
        self._type, self._element = tokens
        #print(self._type)
        self._cstr = None

    @property
    def type(self):
        return self.get_type_defining_node().type
    
    @property
    def kind(self):
        return self.get_type_defining_node().kind
    
    @property
    def bytes_per_element(self):
        return self.get_type_defining_node().bytes_per_element

    @property
    def rank(self):
        return self.get_rank_defining_node().rank
    
    @property
    def symbol_info(self):
        return self._value.symbol_info

    def child_nodes(self):
        yield self._type
        yield self._element

    def walk_derived_type_parts_preorder(self):
        """Yields all TTDerivedTypePart instances.
        """
        yield self
        if isinstance(self._element,TTDerivedTypePart):
            yield from self._element.walk_derived_type_parts_preorder()
    
    def walk_derived_type_parts_postorder(self):
        """Yields all TTDerivedTypePart instances.
        """
        if isinstance(self._element,TTDerivedTypePart):
            yield from self._element.walk_derived_type_parts_postorder()
        yield self

    def get_innermost_member(self):
        """:return: inner most derived type member.
        """
        for current in self.walk_derived_type_parts_preorder():
            pass
        return current

    def get_type_defininig_node(self):
        return self.get_innermost_member()._element

    def get_rank_defining_node(self):
        r""":return: The subexpression part that contains the information on 
        the rank of this derived type member access expression.
        :note: Assumes semantics check has been performed beforehand
        so that we can assume that not more than one part of the derived
        type member access expression has a rank greater than 0.

        **Examples:**

        arr1(:)%scal1 -> returns arr1
        arr1(i)%scal1 -> returns scal2
        arr1(i)%arr2 -> returns arr2
        arr1%arr2(i) -> returns arr1
        """
        for current in self.walk_derived_type_parts_preorder():
            if current._type.rank > 0:
                return current._type
        return current._element 

    @property
    def is_array(self):
        """:note: Does not require array to be contiguous.
                  Expressions like array(:)%scalar are thus 
                  allowed too. Hence we use the rank-defining 
                  derived type part.
        """
        return self.get_rank_defining_node().is_array
        
    @property
    def is_contiguous_array(self):
        """note: Only applicable to innermost element.
                 hence we use the innermost, the type-defining, 
                 derived type part."""
        return self.get_type_defining_node().is_contiguous_array
    
    @property
    def is_full_array(self):
        """note: Only applicable to innermost element.
                 hence we use the innermost, the type-defining, 
                 derived type part."""
        return self.get_type_defining_node().is_full_array
    
    @property
    def is_scalar(self):
        return self.get_type_defining_node().is_scalar

    @property
    def args(self):
        if isinstance(self._value,TTFunctionCall):
            yield from self._value.args()
        else:
            yield from ()

    def slice_args(self):
        """Returns all range args in the order of their appeareance.
        """
        if isinstance(self._value,TTFunctionCall):
            yield from self._value.slice_args()
        else:
            yield from ()

    def identifier_part(self,converter=traversals.make_fstr):
        # todo: adopt walk_members generator
        result = converter(self._type)
        current = self._element
        while isinstance(current,TTDerivedTypePart):
            current = current._element
            result += "%"+converter(self._type)
        if isinstance(current,TTFunctionCall):
            result += "%"+converter(current._name)
        else: # TTIdentifier
            result += "%"+converter(current)
        return result             

    def overwrite_cstr(self,expr):
        self._cstr = expr

    def cstr(self):
        if self._cstr == None:
            return traversals.make_cstr(self._type) + "." + traversals.make_cstr(
                self._element)
        else:
            return self._cstr

    def fstr(self):
        return traversals.make_fstr(self._type) + "%" + traversals.make_fstr(
            self._element)
    
    def __str__(self):
        return "TTDerivedTypePart(name:"+str(self._type)+"member:"+str(self._element)+")"
    __repr__ = __str__

class TTValue(base.TTNode):
    
    def _assign_fields(self, tokens):
        self._value = tokens[0]
        self._reduction_index = None
        self._fstr = None

    def get_type_defining_node(self):
        if isinstance(self._value,(TTDerivedTypePart)):
            return self._value.get_type_defininig_node()
        else:
            return self._value

    def get_rank_defining_node(self):
        if isinstance(self._value,(TTDerivedTypePart)):
            return self._value.get_rank_defining_node()
        else:
            return self._value

    @property
    def type(self):
        return self.get_type_defining_node().type
    
    @property
    def kind(self):
        return self.get_type_defining_node().kind
    
    @property
    def bytes_per_element(self):
        return self.get_type_defining_node().bytes_per_element

    @property
    def rank(self):
        return self.get_rank_defining_node().rank

    def child_nodes(self):
        yield self._value
   
    def is_identifier(self):
        return isinstance(self._value, TTIdentifier)

    def identifier_part(self,converter=traversals.make_fstr):
        """
        :return: The identifier part of the expression. In case
                 of a function call/tensor access expression, 
                 excludes the argument list.
                 In case of a derived type, excludes the argument
                 list of the innermost member.
        """
        if type(self._value) is TTFunctionCall:
            return converter(self._value._name)
        elif type(self._value) is TTDerivedTypePart:
            return self._value.identifier_part(converter)
        else:
            return converter(self._value)
    
    @property
    def is_array(self):
        return self.get_rank_defining_node().is_array
    
    @property
    def is_scalar(self):
        return self.get_rank_defining_node().is_scalar
    
    @property
    def is_contiguous_array(self):
        return self.get_type_defining_node().is_contiguous_array
    
    @property
    def is_full_array(self):
        return self.get_type_defining_node().is_full_array
    
    def is_function_call(self):
        """:note: TTFunctionCall instances must be flagged as tensor beforehand.
        """
        # todo: check if detect all arrays in indexer/scope
        # so that we do not need to know function names anymore.
        if type(self._value) is TTFunctionCall:
            return self._value.is_function_call()
        elif type(self._value) is TTDerivedTypePart:
            # todo: support type bounds routines
            return False
        else:
            return False

    def get_value(self):
        return self._value 

    def name(self):
        return self._value.fstr()

    def slice_args(self):
        if type(self._value) is TTFunctionCall:
            yield from self._value.slice_args()
        elif type(self._value) is TTDerivedTypePart:
            yield from self._value.innermost_member_slice_args()
        else:
            yield from ()
   
    @property    
    def args(self):
        if type(self._value) is TTFunctionCall:
            yield from self._value._args
        elif type(self._value) is TTDerivedTypePart:
            yield from self._value.innermost_member_args()
        else:
            yield from ()
    
    def fstr(self):
        if self._fstr != None:
            return self._fstr
        else:
            return traversals.make_fstr(self._value)

    def cstr(self):
        result = traversals.make_cstr(self._value)
        if self._reduction_index != None:
            if opts.fortran_style_array_access:
                result += "({idx})".format(idx=self._reduction_index)
            else:
                result += "[{idx}]".format(idx=self._reduction_index)
        return result.lower()

class TTLvalue(TTValue):
    def __str__(self):
        return "TTLvalue(val:"+str(self._value)+")"
    __repr__ = __str__

class TTRvalue(TTValue):
    def __str__(self):
        return "TTRvalue(val:"+str(self._value)+")"
    __repr__ = __str__

def _need_to_add_brackets(op,other_opd):
    return isinstance(other_opd,(TTUnaryOp,TTBinaryOpChain))

class OperatorType(enum.Enum):
    UNKNOWN = 0
    POW = 1 # only applicable to number types (integer,real,complex)
    ADD = 2 # only applicable to number types (integer,real,complex)
    MUL = 3 # only applicable to number types (integer,real,complex)
    COMP = 4 # only applicable to number types (integer,real,complex)
    LOGIC = 5 # only applicable to logical types

class TTUnaryOp(base.TTNode):
    f2c = {
      ".not.": "!{r}",
      "+": "{r}",
      "-": "-{r}",
    }
    
    def _assign_fields(self, tokens):
        self.op, self.opd = tokens[0]
        self._op_type = None
    
    def operator_type(self):
        """:return: a characteristic operator for 
        the binary operators aggregated in this chain
        of binary operations.
        """
        if self._op_type == None:
            op = self.operators[0].lower()
            if op in ["+","-"]:
                self._op_type = OperatorType.ADD
            else:
                self._op_type =  OperatorType.LOGIC
        return self._op_type 
    
    def child_nodes(self):
        yield self.opd
    
    @property
    def type(self):
        return self.opd.type
    
    @property
    def bytes_per_element(self):
        return self.opd.bytes_per_element

    @property
    def rank(self):
        return self.opd.rank
 
    def _op_c_template(self):
        return TTUnaryOp.f2c.get(self.op.lower())
    def cstr(self):
        if _need_to_add_brackets(self,opd):
            return self._op_c_template().format(
              r="(" + self.opd.cstr() + ")"
            )
        else:
            return self._op_c_template().format(
              r=self.opd.cstr()
            )
    
    def fstr(self):
        if _need_to_add_brackets(self,opd):
            return self.op + "(" + self.opd.fstr() + ")"
        else:
            return self.op + self.opd.fstr()

class TTBinaryOpChain(base.TTNode):
    """pyparsing's infixNotation flattens
    repeated applications of binary
    operator expressions that have the same precedence 
    into a single list of tokens.
    This class models such lists of grouped operator
    and operand tokens.

    Example:
    
    As '+' and '-' have the same precendence, 
    parsing the expression
    `a + b + c - d`
    will have pyparsing's infixNotation group the
    tokens as follows: 
    ['a','+','b','+','c','-','d'].
    While it parses the expression `a + b - c*d`
    as below:
    ['a','+','b','-',['c','*','d']]
    """
    
    f2c = {
      "**":"__pow({l},{r})",
      "*": "{l} * {r}",
      "/": "{l} / {r}",
      "+": "{l} + {r}",
      "-": "{l} - {r}",
      "<": "{l} < {r}",
      "<=": "{l} <= {r}",
      ">": "{l} > {r}",
      ">=": "{l} >= {r}",
      ".lt.": "{l} < {r}",
      ".le.": "{l} <= {r}",
      ".gt.": "{l} > {r}",
      ".ge.": "{l} >= {r}",
      "==": "{l} == {r}",
      "/=": "{l} != {r}",
      ".and.": "{l} && {r}",
      ".or.":  "{l} || {r}",
      ".eq.": "{l} == {r}",
      ".eqv.": "{l} == {r}",
      ".ne.": "{l} != {r}",
      ".neqv.": "{l} != {r}",
      ".xor.": "{l} ^ {r}",
    }
    
    def _assign_fields(self, tokens):
        #print([type(tk) for tk in tokens[0]])
        self.exprs = tokens[0]
        self._op_type = None
        self._type = None
        self._kind = None
        self._rank = None
        self._bytes_per_element = None

    @property
    def operator_type(self):
        """:return: a characteristic operator for 
        the binary operators aggregated in this chain
        of binary operations.
        
        Returns '&&' for all logical comparisons,
        '==' for all number comparisons, '*' for [*/]
        and '+' for [+-].
        """
        if self._op_type == None:
            op = self.operators[0].lower()
            if op == "**":
                self._op_type = OperatorType.POW
            elif op in ["*","/"]:
                self._op_type = OperatorType.MUL
            elif op in ["+","-"]:
                self._op_type = OperatorType.ADD
            elif op in [
              "<",
              "<=",
              ">",
              ">=",
              ".lt.",
              ".le.",
              ".gt.",
              ".ge.",
              "==",
              "/=",
            ]:
                self._op_type =  OperatorType.COMP
            elif op in [
              ".and.",
              ".or.",
              ".eq.",
              ".eqv.",
              ".ne.",
              ".neqv.",
              ".xor.",
            ]:
                self._op_type =  OperatorType.LOGIC
        return self._op_type 
    
    @property
    def operands(self):
        return self.exprs[::2]

    @property
    def operators(self):
        return self.exprs[1::2]
    
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
        for opd in self.operands:
            yield opd
    
    def _op_c_template(self,op):
        return TTBinaryOpChain.f2c.get(op.lower())
    
    def cstr(self):
        # a + b + c
        # + + 
        # a b c
        # add(add(a,b),c)
        opds = self.operands
        lopd = opds[0]
        result = lopd.cstr()
        if _need_to_add_brackets(self,lopd): 
            result="(" + result + ")"
        for i,op in enumerate(self.operators):
            op_template = self._op_c_template(op)
            l = result
            ropd = opds[i+1]
            r = ropd.cstr()
            if _need_to_add_brackets(self,ropd): 
                r="(" + r + ")"
            result = op_template.format(
              l = result,
              r = r
            )
        return result
    
    def fstr(self):
        opds = self.operands
        result = opds[0].fstr()
        if _need_to_add_brackets(self,opds[0]): 
            result="(" + result + ")"
        for i,op in enumerate(self.operators):
            op_template = "{l} {op} {r}"
            l = result
            r = opds[i+1].fstr()
            if _need_to_add_brackets(self,opds[i+1]): 
                r="(" + r + ")"
            result = op_template.format(
              l = l,
              r = r,
              op = self.op
            )
        return result

class TTArithExpr(base.TTNode):

    def _assign_fields(self, tokens):
        self._expr = tokens[0] # either: rvalue,unary op,binary op

    def child_nodes(self):
        yield self._expr
    
    @property
    def type(self):
        return self._expr.type

    @property
    def rank(self):
        return self._expr.rank
    
    @property
    def bytes_per_element(self):
        return self._expr.bytes_per_element
    
    @property
    def is_array(self):
        return self._expr.is_array

    @property
    def is_contiguous_array(self):
        return self._expr.is_contiguous_array
    
    @property
    def is_full_array(self):
        return self._expr.is_full_array
    
    @property
    def is_scalar(self):
        return self._expr.is_scalar

    def walk_rvalues_preorder(self):
        """Yields all TTDerivedTypePart instances.
        """
        yield self
        if isinstance(self._element,TTDerivedTypePart):
            yield from self._element.walk_derived_type_parts_preorder()
    
    def walk_rvalues_postorder(self):
        """Yields all TTDerivedTypePart instances.
        """
        if isinstance(self._expr,TTRvalue):
            yield self._expr
        
    def cstr(self):
        return self._expr.cstr()

    def fstr(self):
        return self._expr.fstr()

class TTComplexArithExpr(base.TTNode):

    def _assign_fields(self, tokens):
        self._real, self._imag = tokens[0]

    def child_nodes(self):
        yield self._real; yield self._imag

    @property
    def type(self):
        return Literal.REAL

    @property
    def kind(self):
        if self.bytes_per_element == 4:
            return "c_float_complex"
        elif self.bytes_per_element == 8:
            return "c_double_complex"
        elif self.bytes_per_element == 16:
            return "c_long_double_complex"
        else:
            assert False, "not supported"

    @property
    def rank(self):
        return 0

    @property
    def bytes_per_element(self):
        """:return: Size of the real/imag component of this complex arithmetic expression.
        - If one of the components is a 16-byte real, the result is a 32 byte complex (2 quads)
        - Else:
          - If one of the components is an 8-byte real, the result is a 16 byte complex (2 doubles)
          - Else:
              the result is a 8-byte complex (2 floats)
        """
        max_bytes_per_element = 4
        for component in self.child_nodes():
           if component.type == Literal.REAL:
               max_bytes_per_element = max(
                 max_bytes_per_element,
                 component.bytes_per_element
               )
        return max_bytes_per_element

    def cstr(self):
        return "make_hip_complex({real},{imag})".format(\
                real=traversals.make_cstr(self._real),\
                imag=traversals.make_cstr(self._imag))

    def fstr(self):
        return "({real},{imag})".format(\
                real=traversals.make_fstr(self._real),\
                imag=traversals.make_fstr(self._imag))

class TTKeywordArgument(base.TTNode):
    def _assign_fields(self, tokens):
        self.key, self.value = tokens

    @property
    def type(self):
        return self.value.type
    
    @property
    def kind(self):
        return self.value.kind
    
    @property
    def bytes_per_element(self):
        return self.value.bytes_per_element

    @property
    def rank(self):
        return self.value.rank

    def child_nodes(self):
        yield self.key; yield self.value
    def cstr(self):
        return self._lhs.cstr() + "=" + self._rhs.cstr() + ";\n"
    def fstr(self):
        return self._lhs.fstr() + "=" + self._rhs.fstr() + ";\n"

class TTSlice(base.TTNode):

    def _assign_fields(self, tokens):
        self._lbound, self._ubound, self._stride =\
          None, None, None 
        if len(tokens) == 1:
            self._ubound = tokens[0]
        elif len(tokens) == 2:
            self._lbound = tokens[0]
            self._ubound = tokens[1]
        elif len(tokens) == 3:
            self._lbound = tokens[0]
            self._ubound = tokens[1]
            self._stride = tokens[2]

    @property
    def type(self):
        return self._ubound.type
    @property
    def kind(self):
        return self._ubound.kind
    @property
    def bytes_per_element(self):
        return self._ubound.bytes_per_element
    @property
    def rank(self):
        return self._ubound.rank

    def has_lbound(self):
        return self._lbound != None
    def has_ubound(self):
        return self._ubound != None
    def has_stride(self):
        return self._stride != None

    def l_bound(self, converter=traversals.make_cstr):
        return converter(self._lbound)
    def u_bound(self, converter=traversals.make_cstr):
        return converter(self._ubound)
    def stride(self, converter=traversals.make_cstr):
        return converter(self._stride)

    def set_loop_var(self, name):
        self._loop_var = name

    def size(self, converter=traversals.make_cstr):
        result = "{1} - ({0}) + 1".format(converter(self._lbound),
                                          converter(self._ubound))
        try:
            result = str(ast.literal_eval(result))
        except:
            try:
                result = " - ({0}) + 1".format(converter(self._lbound))
                result = str(ast.literal_eval(result))
                if result == "0":
                    result = converter(self._ubound)
                else:
                    result = "{1}{0}".format(result, converter(self._ubound))
            except:
                pass
        return result

    def cstr(self):
        #return self.size(traversals.make_cstr)
        return "/*TODO fix this BEGIN*/{0}/*fix END*/".format(self.fstr())

    def overwrite_fstr(self,fstr):
        self._fstr = fstr
    
    def fstr(self):
        if self._fstr != None:
            return self._fstr
        else:
            result = ""
            if not self._lbound is None:
                result += traversals.make_fstr(self._lbound)
            result += ":"
            if not self._ubound is None:
                result += traversals.make_fstr(self._ubound)
            if not self._stride is None:
                result += ":" + traversals.make_fstr(self._stride)
            return result

def set_arith_expr_parse_actions(grammar):
    grammar.logical.setParseAction(TTLogical)
    grammar.character.setParseAction(TTCharacter)
    grammar.number.setParseAction(TTNumber)
    grammar.identifier.setParseAction(TTIdentifier)
    grammar.rvalue.setParseAction(TTRvalue)
    grammar.lvalue.setParseAction(TTLvalue)
    grammar.derived_type_elem.setParseAction(TTDerivedTypePart)
    grammar.function_call.setParseAction(TTFunctionCall)
    grammar.tensor_slice.setParseAction(TTSlice)
    grammar.arith_expr.setParseAction(TTArithExpr)
    grammar.complex_arith_expr.setParseAction(
        TTComplexArithExpr)
    grammar.keyword_argument.setParseAction(TTKeywordArgument)
