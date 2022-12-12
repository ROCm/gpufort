# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
"""

Concepts:

Aside from the operators, 

* `is_<tensor_kind>` indicate that the expression is actually of the tensor kind (scalar, array (contiguous,full,generic)).
* `yield_<tensor_kind>` routines indicate that an operator, including the identity operator,
  as well as elemental and transformational function calls produce a tensor of that kind.

Generators:

* **walk_values_ignore_args**: 

  Certain nodes are equipped with this generator.
  This generator can be used to analyze the operands of 
  Fortran array expressions. Unlike the `walk_preorder` and `walk_postorder`
  routines, the generator does not descend into function call arguments and array indices.
  The only exception being elemental function calls.
  Here, the generator descends directly into the argument list and does
  not yield the elemental function call expression itself.
  It further does not yield any unary and binary operators. 
  It exclusively yields TTValue instances.

* **walk_variable_references**: 
    
  Walks across all nodes and yields those that are linked one-to-one 
  to entries in the symbol table. For example, given an expression
  `a%b%c(i)` this generator would yield the nodes (TTIdentifier,TTFunctionCall) 
  representing `a`,`a%b`, and `a%b%c` and `i`.
                            
"""

import pyparsing
import ast

from gpufort import util
from gpufort import indexer

from .. import opts
from .. import conv
from .. import prepostprocess

from . import base
from . import traversals

import enum

# literal expressions
    
class FortranType:
    """Enum-like class containing constants 
    for Fortran types."""
    LOGICAL = "logical"
    CHARACTER = "character"
    INTEGER = "integer"
    REAL = "real"
    COMPLEX = "complex"
    TYPE = "type"

class ArithExprNode(base.TTNode):
    """Common base clase for the nodes that constitute
    an arithmetic expression.
    :note: Assumes that all subclasses have a 'type' property.
    """
    
    @property
    def yields_integer(self):
        """:return: If the expression yields an 'integer' type.
        """
        return self.type == FortranType.INTEGER
    
    @property
    def yields_real(self):
        """:return: If the expression yields an 'real' type.
        """
        return self.type == FortranType.REAL
    
    @property
    def yields_complex(self):
        """:return: If the expression yields an 'complex' type.
        """
        return self.type == FortranType.COMPLEX
    
    @property
    def yields_logical(self):
        """:return: if the expression yields an 'logical' type.
        """
        return self.type == FortranType.LOGICAL

    @property
    def yields_character_type(self):
        """:return: If the expression yields an 'character' type.
        """
        return self.type == FortranType.CHARACTER

    @property
    def yields_derived_type(self):
        return self.type == FortranType.TYPE
    
    @property
    def yields_numeric_type(self):
        """:return: If the expression yields an 'integer','real', or 'complex' type.
        :note: No kind information considered in this check.
        """
        return self.type in [
          FortranType.INTEGER,
          FortranType.REAL,
          FortranType.COMPLEX,
        ]
        

class Literal(ArithExprNode):
            
    def _assign_fields(self, tokens):
        """
        Expected inputs for 
        """
        self._raw_value = tokens[0]
        self._value = self._raw_value
        self._kind = None
        if "_" in self._raw_value:
            parts = self._raw_value.split("_",maxsplit=1)
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
    def rank(self): return 0
    
    @property
    def is_scalar(self): return True
    @property
    def is_array(self): return False
    @property
    def is_contiguous_array(self): return False
    @property
    def is_full_array(self): return False
    
    @property
    def yields_scalar(self): return self.is_scalar
    @property
    def yields_array(self): return self.is_array
    @property
    def yields_contiguous_array(self): return self.is_contiguous_array
    @property
    def yields_full_array(self): return self.is_full_array
    
    def walk_variable_references(self):
        yield from ()

    @property
    def resolved(self):
        return self._bytes_per_element != None
    
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
        return FortranType.CHARACTER 
   
    @property
    def len(self):
        return len(self._raw_value)-2
 
    def cstr(self):
        return '"'+self._raw_value[1:-1]+'"'

class TTLogical(Literal):

    @property
    def type(self):
        return FortranType.LOGICAL

    def cstr(self):
        if self.bytes_per_element == "1":
            return "true" if self._value.lower() == ".true." else "false"
        else:
            return "1" if self._value.lower() == ".true." else "0"

class TTNumber(Literal):

    def _assign_fields(self, tokens):
        Literal._assign_fields(self,tokens)
        if "." in self._value:
            self._type = FortranType.REAL 
            if self._kind == None:
                if "d" in self._value:
                    self._kind = "c_double"
        else:
            self._type = FortranType.INTEGER
    
    @property
    def type(self):
        return self._type

    def cstr(self):
        # todo: check kind parameter in new semantics part
        if self._type == FortranType.REAL:
            if self.bytes_per_element == 4:
                return self._value + "f"
            elif self.bytes_per_element == 8:
                return self._value.replace("d", "e")
            else:
                raise util.error.LimitationError("only single & double precision floats supported")
        elif self._type == FortranType.INTEGER:
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

class VarExpr(ArithExprNode):

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
        return self._get_type_defining_record().type
    
    @property 
    def is_parameter(self):
        return self._get_type_defining_record().is_parameter

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

class TTIdentifier(VarExpr):
        
    def _assign_fields(self, tokens):
        self.name = tokens[0]
        VarExpr._assign_fields(self,tokens)
    
    def walk_variable_references(self):
        yield self

    @property 
    def rank(self):
        assert self.partially_resolved
        return self.symbol_info.rank
    
    @property 
    def kind(self):
        return self._get_type_defining_record().kind
    
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

    @property
    def yields_array(self):
        return self.rank > 0
    @property
    def yields_contiguous_array(self):
        return self.yields_array
    @property
    def yields_full_array(self):
        return self.yields_array
    @property
    def yields_scalar(self):
        return not self.yields_array
  
    def fstr(self):
        return self.name

    def cstr(self):
        return self.name.lower()

    def __str__(self):    
        return "TTIdentifier(name:"+str(self._name)+")"
    __repr__ = __str__


class TTFunctionCall(VarExpr):
    """Tree node for function call and array access expressions. In Fortran,
    generally the lookup table need to be queried
    in order to identify the actual type of an
    expression that appears like a function call.

    After the semantic analysis, one can 
    use the properties 'is_function_call` and
    `is_array_expr` to check what this node actually represents.
    """ 
 
    def _assign_fields(self, tokens):
        self.name = tokens[0]
        if len(tokens) > 1:
            self.args = tokens[1:]
        else:
            self.args = []
        VarExpr._assign_fields(self,tokens)
    
    # override
    def _get_type_defining_record(self):
        symbol_info = self.symbol_info
        if isinstance(symbol_info,indexer.indexertypes.IndexVariable): # is variable
            return symbol_info
        else:
            assert isinstance(symbol_info,indexer.indexertypes.IndexProcedure)
            result_name = symbol_info.result_name
            return symbol_info.get_variable(result_name) 
    
    @property 
    def kind(self):
        """In case of function call, if result depends on `kind` argument, 
        checks for `kind` argument and returns expression.
        If `kind` argument not found, returns `None` (default kind).
        :note: Assumes semantics have been analyzed before.
        """
        if (self.is_function_call 
            and self.result_depends_on_kind):
            try:
                arg = self.get_actual_argument("kind")
                if isinstance(arg,TTKeywordArgument):
                    return arg.value.fstr()
                else:
                    return arg.fstr()
            except util.error.LookupError:
                return None # default
        else:
            return self._get_type_defining_record().kind

    @property
    def rank(self):
        """:note: Assumes semantics have been checked before.
        """
        if ( self.is_function_call 
             and self.is_elemental_function_call ):
              max_rank = 0
              for arg in self.args:
                  max_rank = max(max_rank,arg.rank) 
              return max_rank
        elif self.is_array_expr:
            return len([arg for arg in self.index_range_args()])
        else: 
            return self._get_type_defining_record().rank

    def child_nodes(self):
        for arg in self.args:
            yield arg

    def walk_variable_references(self):
        """Yields itself if self is a variable.
        Then yields from arguments."""
        if self.is_array_expr:
            yield self
        for arg in self.args:
            yield from arg.walk_variable_references()

    def index_range_args(self):
        """Returns all range args in the order of their appeareance.
        """
        # todo: double check that this does not descend
        for arg in self.args:
            if isinstance(arg,TTIndexRange):
                yield arg

    @property
    def is_array_expr(self):
        """:return: if this expression represents
        an array (full array or subarray) or a single array element.
        :see: is_subarray.
        :see: is_full_aray"""
        return isinstance(self.symbol_info,indexer.indexertypes.IndexVariable)
 
   
    @property
    def is_subarray(self):
        """:return: if this is a subarray expression,
        i.e. if it contains index ranges."""
        if self.is_array_expr:
            for arg in self.args:
                if isinstance(arg,TTIndexRange):
                    return True
        return False
 
    @property
    def is_array(self):
        """:return: if the expression describes
        an array.
        Might be a subarray, a full array but no array element.
        :see: is_subarray
        :see is_full_array"""
        return self.is_subarray
    
    @property
    def is_scalar(self):
        """Checks if none of the array index arguments 
        are index ranges.
        """
        return not self.is_array
    
    @property
    def is_contiguous_array(self):
        """Checks if this array section expression
        is a contiguous section of an original array.
        
        :return: true if one or more index range have been found at 
          the begin of the index argument list
            - and all bounds and step sizes are unspecified with
              the only allowed exception being the lower and upper bound of the last index range.
        
        :note: It is generally not possible to know by static analysis
               if an array argument to a procedure is contiguous 
               if this is not indicated via the `contiguous` keyword.
               On the device, we have to assume
               that all array arguments that are mapped to the device are contiguous.
               We need to outsource the check of this property to the host compiler.
        :todo: Currently assumes that array contiguity is destroyed if
               a lower bound is specified for any index range at position != 0.
               However, we might need to consider scenarios where an index lower bound 
               is specified and coincides with the array variable's lower bound.
        """
        if self.is_array_expr:
            last_index_range = -1 # must be initialized with -1
                            # as then: i-last==1 at i=0
            first_index_range_with_ubound_or_lbound = 10**20
            for i,arg in enumerate(self.args):
                if isinstance(arg,TTIndexRange):
                    if (
                        arg.has_stride()
                        or i - last_index_range > 1 # distance between strides > 1
                        or i > first_index_range_with_ubound_or_lbound
                      ):
                        return False
                    if (arg.has_lbound()
                       or arg.has_ubound()):
                        first_index_range_with_ubound_or_lbound = min(
                          i,first_index_range_with_ubound_or_lbound
                        )
                    last_index_range = i
            # no strides in any index_range at this point
            return last_index_range >= 0
        else:
            return False

    @property
    def is_full_array(self):
        for arg in self.args:
            if (not isinstance(arg,TTIndexRange)
               or arg.has_lbound()
               or arg.has_ubound()
               or arg.has_stride()):
                return False
        return True
 
    @property
    def yields_array(self):
        """If this an array expression, check if there is an index range.
        If this is an elemental function, check if one argument
        is an array. If this is any other function, check if the result
        is an array.
        """
        # todo incomplete
        if self.is_function_call:
           if self.is_elemental_function_call:
               for arg in self.args:
                   if arg.yields_array:
                       return True
           else:
                return self.symbol_info.result.rank > 0
        elif self.is_array_expr:
            return self.is_array 
        return False
       
    def iter_actual_arguments(self):
        """Yields a tuple of argument name
        and value expression.
        :note: Can only be used safely after semantics analysis.""" 
        for pos,arg in enumerate(self.args):
            if isinstance(arg,TTKeywordArgument):
                name = arg.key.lower()
                value = arg.value
            else:
                name = self.dummy_args[pos]
                value = arg
            yield (name, value)
 
    @property
    def yields_contiguous_array(self):
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
        if self.is_function_call:
            if self.is_elemental_function_call:
                one_array_is_contiguous = False
                for name, value in self.iter_actual_arguments():
                    if value.yields_contiguous_array:
                        one_array_is_contiguous = True
                    if not (
                        value.yields_contiguous_array
                        or value.yields_scalar
                      ):
                        if self.result_depends_on_kind:
                            if name != "kind":
                                return False
                        else:
                            return False
                return one_array_is_contiguous
            else:
                return self.symbol_info.result.rank > 0
        elif self.is_array_expr:
            return self.is_contiguous_array
        else:
            return False

    @property
    def yields_full_array(self):
        """Checks if this array section expression
        covers all elements of an original array.
        For elemental functions
        """
        if self.is_function_call:
           if self.is_elemental_function_call:
                one_array_is_full = False
                for name, value in self.iter_actual_arguments():
                    if value.yields_full_array:
                        one_array_is_full = True
                    if not (
                        value.yields_full_array
                        or value.yields_scalar
                      ):
                        if self.result_depends_on_kind:
                            if name != "kind":
                                return False
                        else:
                            return False
                return one_array_is_full
           else:
               return self.symbol_info.result.rank > 0
        elif self.is_array_expr:
            return self.is_full_array
        else:
            return False
    
    @property
    def yields_scalar(self):
        """Checks if this expression yields a scalar. 
        """
        return not self.yields_array

    @property
    def is_function_call(self):
        return not self.is_array_expr
    
    @property
    def is_intrinsic_call(self):
        """This is a call to a Fortran intrinsic."""
        assert self.is_function_call
        return self.symbol_info.is_intrinsic
   
    @property
    def is_elemental_function_call(self):
        """This is a call to an elemental function."""
        assert self.is_function_call
        return self.symbol_info.is_elemental

    @property
    def is_converter_call(self):
        """:note: Conversions are always elemental."""
        assert self.is_elemental_function_call
        return self.symbol_info.is_conversion 
    
    @property
    def result_depends_on_kind(self):
        """Result of the function depends on a kind argument."""
        assert self.is_elemental_function_call
        return self.symbol_info.result_depends_on_kind

    @property
    def dummy_args(self):
        """Names of arguments in their expected order
        when all arguments are supplied as positional arguments.
        """
        assert self.is_function_call
        return self.symbol_info.dummy_args

    def get_expected_argument(self,arg_name):
        assert self.is_function_call
        return self.symbol_info.get_argument(arg_name)

    def get_actual_argument(self,arg_name):
        """:note: Assumes semantics have been analyzed before.
        :throws: util.error.LookupError
        """
        assert self.is_function_call
        arg_name = arg_name.lower()
        for i,arg in enumerate(self.args):
            if isinstance(arg,TTKeywordArgument):
                if arg.key.lower() == arg_name:
                    return arg
            elif self.dummy_args[i] == arg_name:
                  return arg
        raise util.error.LookupError(
          "no argument found for name '{}'".format(
            arg_name
          )
        )

    def is_argument_specified(self,arg_name):
        """:return: If the given argument is specified."""
        assert self.is_function_call
        arg_name = arg_name.lower()
        assert arg_name in self.symbol_info.dummy_args
        try:
            self.get_actual_argument(arg_name)
            return True
        except util.error.LookupError:
            return False

    def _args_as_str(self,converter):
        """Assumes Fortran style () operators are used in any case.
        :note: This is the default implementation."""
        args = self.args
        if len(args):
            return "({0})".format(",".join(
                converter(el) for el in args))
        else:
            return ""

    def cstr(self):
        return (
          self.name.lower()
          + self._args_as_str(
            traversals.make_cstr
          )
        )
    
    def fstr(self):
        return (
          self.name
          + self._args_as_str(
            traversals.make_fstr
          )
        )
    
    def __str__(self):
        return "TTFunctionCall(name:"+str(self.name)+",is_array_expr:"+str(self.is_array_expr)+")"
    __repr__ = __str__

class TTIndexRange(ArithExprNode):

    def _assign_fields(self, tokens):
        self.lbound, self.ubound, self.stride =\
          None, None, None  # note: all three might be None
        if len(tokens) == 1:
            self.ubound = tokens[0]
        elif len(tokens) == 2:
            self.lbound = tokens[0]
            self.ubound = tokens[1]
        elif len(tokens) == 3:
            self.lbound = tokens[0]
            self.ubound = tokens[1]
            self.stride = tokens[2]
        else: # ::
            assert len(tokens) == 0
        self._type = None
        self._bytes_per_element = None

    def child_nodes(self):
        if self.lbound != None:
            yield self.lbound
        if self.ubound != None:
            yield self.ubound
        if self.stride != None:
            yield self.stride
    
    @property
    def type(self):
        assert self._type != None
        return self._type
    @type.setter
    def type(self,typ):
        self._type = typ
 
    @property
    def bytes_per_element(self):
        assert self._bytes_per_element != None
        return self._bytes_per_element
    @bytes_per_element.setter
    def bytes_per_element(self,bytes_per_element):
        self._bytes_per_element = bytes_per_element

    @property
    def rank(self):
        return 1

    def has_lbound(self):
        return self.lbound != None
    def has_ubound(self):
        return self.ubound != None
    def has_stride(self):
        return self.stride != None
    
    @property
    def is_scalar(self): return False
    @property
    def is_array(self): return True
    @property
    def is_contiguous_array(self): return not self.has_stride # :todo: support unit strides 
    @property
    def is_full_array(self): return self.is_contiguous_array # :todo: cannot know the bounds of array, must be set during semantic analysis
    
    @property
    def yields_scalar(self): return self.is_scalar
    @property
    def yields_array(self): return self.is_array
    @property
    def yields_contiguous_array(self): return self.is_contiguous_array
    @property
    def yields_full_array(self): return self.is_full_array

    def size(self, converter=traversals.make_cstr):
        result = "{1} - ({0}) + 1".format(
          converter(self.lbound),
          converter(self.ubound)
        )
        try:
            result = str(ast.literal_eval(result))
        except:
            try:
                result = " - ({0}) + 1".format(converter(self.lbound))
                result = str(ast.literal_eval(result))
                if result == "0":
                    result = converter(self.ubound)
                else:
                    result = "{1}{0}".format(result, converter(self.ubound))
            except:
                pass
        return result

    def cstr(self):
        #return self.size(traversals.make_cstr)
        return "/*TODO fix this BEGIN*/{0}/*fix END*/".format(self.fstr())
    
    def fstr(self):
        result = ""
        if not self.lbound is None:
            result += traversals.make_fstr(self.lbound)
        result += ":"
        if not self.ubound is None:
            result += traversals.make_fstr(self.ubound)
        if not self.stride is None:
            result += ":" + traversals.make_fstr(self.stride)
        return result

class TTDerivedTypePart(ArithExprNode):

    def _assign_fields(self, tokens):
        self._derived_type, self._element = tokens
        #print(self._type)
        self._cstr = None

    @property
    def derived_type(self):
        return self._derived_type
    
    @property
    def element(self):
        return self._element

    def child_nodes(self):
        yield self._derived_type
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
    
    def walk_variable_references(self):
        """Yields all variables referenced by this expression in preorder.
        """
        yield self._derived_type
        yield from self._element.walk_variable_references()

    @property
    def innermost_part(self):
        """:return: inner most derived type part.
        """
        for current in self.walk_derived_type_parts_preorder():
            pass
        return current

    @property
    def type_defininig_node(self):
        return self.innermost_part.element

    @property
    def rank_defining_node(self):
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
            if current.derived_type.rank > 0:
                return current.derived_type
        return current.element 

    @property
    def name(self):
        result = converter(self._derived_type)
        current = self._element
        while isinstance(current,TTDerivedTypePart):
            current = current.element
            result += "%"+converter(current.derived_type)
        if isinstance(current,TTFunctionCall):
            result += "%"+converter(current.name)
        else: # TTIdentifier
            result += "%"+converter(current)
        return result             

    def cstr(self):
        return traversals.make_cstr(self._derived_type) + "." + traversals.make_cstr(
            self._element)

    def fstr(self):
        return traversals.make_fstr(self._derived_type) + "%" + traversals.make_fstr(
            self._element)
    
    def __str__(self):
        return "TTDerivedTypePart(name:"+str(self._derived_type)+"member:"+str(self._element)+")"
    __repr__ = __str__

class TTComplexConstructor(ArithExprNode):

    def _assign_fields(self, tokens):
        self.real, self.imag = tokens[0]

    def child_nodes(self):
        yield self.real; yield self.imag

    @property
    def type(self):
        return FortranType.COMPLEX

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
    def is_scalar(self): return True
    @property
    def is_array(self): return False
    @property
    def is_contiguous_array(self): return False
    @property
    def is_full_array(self): return False
    
    @property
    def yields_scalar(self): return self.is_scalar
    @property
    def yields_array(self): return self.is_array
    @property
    def yields_contiguous_array(self): return self.is_contiguous_array
    @property
    def yields_full_array(self): return self.is_full_array
    
    def walk_values_ignore_args(self):
        yield from self.real.walk_values_ignore_args()
        yield from self.imag.walk_values_ignore_args()

    @property
    def bytes_per_element(self):
        """:return: Size of the real/imag component of this complex arithmetic expression.
        - If one of the components is a 16-byte real, the result is a 32 byte complex (2 quads)
        - Else:
          - If one of the components is an 8-byte real, the result is a 16 byte complex (2 doubles)
          - Else:
              the result is a 8-byte complex (2 floats)
        """
        max_bytes_per_element = 0
        for component in self.child_nodes():
           if component.type == FortranType.REAL:
               max_bytes_per_element = max(
                 max_bytes_per_element,
                 component.bytes_per_element
               )
        if max_bytes_per_element == 0:
            return 2*4
        else:
            return max_bytes_per_element*2

    def cstr(self):
        return "make_hip_complex({real},{imag})".format(\
                real=traversals.make_cstr(self.real),\
                imag=traversals.make_cstr(self.imag))

    def fstr(self):
        return "({real},{imag})".format(\
                real=traversals.make_fstr(self.real),\
                imag=traversals.make_fstr(self.imag))

class TTArrayConstructor(ArithExprNode):

    def _assign_fields(self, tokens):
        self.entries = tokens

    def child_nodes(self):
        yield from self.entries

    @property
    def type(self):
        """:note: Assumes that semantics check has been performed,
        which ensures that all elements have the same type and size in bytes.
        """
        return self.entries[0].type
    @property
    def bytes_per_element(self):
        """:note: Assumes that semantics check has been performed,
        which ensures that all elements have the same type and size in bytes.
        """
        return self.entries[0].bytes_per_element
    @property
    def rank(self):
        return 1
    @property
    def size(self):
        return len(self.entries)
    
    @property
    def is_scalar(self): return False
    @property
    def is_array(self): return True
    @property
    def is_contiguous_array(self): return True
    @property
    def is_full_array(self): return True
    
    @property
    def yields_scalar(self): return self.is_scalar
    @property
    def yields_array(self): return self.is_array
    @property
    def yields_contiguous_array(self): return self.is_contiguous_array
    @property
    def yields_full_array(self): return self.is_full_array
    
    def walk_values_ignore_args(self):
        yield self

    def cstr(self):
        # todo array types need
        assert False, "Not implemented"

    def fstr(self):
        return "[{}]".format(
          ",".join([traversals.make_fstr(e) for e in self.entries])
        )

class TTValue(ArithExprNode):
    
    def _assign_fields(self, tokens):
        self.value = tokens[0]
        # for overwriting C representation
        self._c_repr_name = None 
        self._c_repr_args = None
        self._c_repr_use_square_brackets = False

    @property
    def type_defining_node(self):
        if isinstance(self.value,(TTDerivedTypePart)):
            return self.value.type_defininig_node
        else:
            return self.value

    @property
    def rank_defining_node(self):
        if isinstance(self.value,(TTDerivedTypePart)):
            return self.value.rank_defining_node
        else:
            return self.value
    
    @property
    def partially_resolved(self):
        if isinstance(self.value,(TTDerivedTypePart)):
            for ttpart in self.value.walk_derived_type_parts_preorder():
                if not ttpart.derived_type.partially_resolved:
                    return False
            return ttpart.element.partially_resolved
        else:
            return self.value.partially_resolved

    @property
    def resolved(self):
        if isinstance(self.value,(TTDerivedTypePart)):
            for ttpart in self.value.walk_derived_type_parts_preorder():
                if not ttpart.derived_type.resolved:
                    return False
            return ttpart.element.resolved
        else:
            return self.value.resolved
    

    @property
    def type(self):
        return self.type_defining_node.type
    
    @property
    def kind(self):
        return self.type_defining_node.kind
    
    @property
    def bytes_per_element(self):
        return self.type_defining_node.bytes_per_element

    @property
    def rank(self):
        return self.rank_defining_node.rank

    def child_nodes(self):
        yield self.value
   
    @property
    def is_identifier_expr(self):
        """The variable expression is a simple identifier."""
        return isinstance(self.value, TTIdentifier)
    @property
    def is_function_call_expr(self):
        """The variable expression is an identifier plus a list of arguments in () brackets.
        :note: Something that looks like a function call in Fortran may actually be 
               an array indexing expression and vice versa. Hence from the syntax alone
               it is often not clear what the expression actually represents.
               In this 
        """
        return isinstance(self.value, TTFunctionCall)
    @property
    def is_derived_type_member_expr(self):
        """The variable expression is a %-delimited sequence of identifiers
        and function call expressions."""
        return isinstance(self.value, TTDerivedTypePart)
    @property
    def is_complex_constructor_expr(self):
        """The variable expression has the shape ( <expr1>, <expr2> ) """
        return isinstance(self.value, TTComplexConstructor)
    @property
    def is_array_constructor_expr(self):
        """The variable expression has the shape [ <expr1>, <expr2>, ... ]
           or [ <expr1>, <expr2>, ... ]."""
        return isinstance(self.value, TTArrayConstructor)

    @property
    def name(self):
        """
        :return: The identifier part of the expression. In case
                 of a function call/tensor access expression, 
                 excludes the argument list.
                 In case of a derived type, excludes the argument
                 list of the innermost member.
        """
        assert isinstance(self.value,
          (TTIdentifier,TTFunctionCall,TTDerivedTypePart)), type(self.value)
        return self.value.name
   
    @property
    def cname(self):
        """C-style identifier of the expression, i.e. '%' replaced by '.'.
        :note: May still contain index range expressions.
        :note: If self._c_repr_name is set, it is returned instead.
        """
        if self._c_repr_name != None:
            return self._c_repr_name.lower()
        else:
            return self.name.replace("%",".").lower()
    
    @property
    def is_array(self):
        """:note: Does not require array to be contiguous.
                  Expressions like array(:)%scalar are thus 
                  allowed too. Hence we use the rank-defining 
                  derived type part.
        """
        return self.rank_defining_node.is_array
    @property
    def is_scalar(self):
        return not self.is_array
    @property
    def is_contiguous_array(self):
        """note: Only applicable to innermost element.
                 hence we use the innermost, the type-defining, 
                 derived type part."""
        return self.type_defining_node.is_contiguous_array
    @property
    def is_full_array(self):
        """note: Only applicable to innermost element.
                 hence we use the innermost, the type-defining, 
                 derived type part."""
        return self.type_defining_node.is_full_array
    # As the innermost node, could be a call to a member
    # function, we cannot simply delegate the yields_<shape> to the is_<shape>
    # functions.
    @property
    def yields_array(self):
        """:note: Does not require array to be contiguous.
                  Expressions like array(:)%scalar are thus 
                  allowed too. Hence we use the rank-defining 
                  derived type part.
        """
        return self.rank_defining_node.yields_array
    @property
    def yields_scalar(self):
        return not self.yields_array
    @property
    def yields_contiguous_array(self):
        """note: Only applicable to innermost element.
                 hence we use the innermost, the type-defining, 
                 derived type part."""
        return self.type_defining_node.yields_contiguous_array
    @property
    def yields_full_array(self):
        """note: Only applicable to innermost element.
                 hence we use the innermost, the type-defining, 
                 derived type part."""
        return self.type_defining_node.yields_full_array
    
    @property
    def is_function_call(self):
        type_defining_node = self.type_defining_node
        if isinstance(type_defining_node,TTFunctionCall):
            return type_defining_node.is_function_call
        else:
            return False

    @property
    def is_intrinsic_call(self):
        type_defining_node = self.type_defining_node
        if isinstance(type_defining_node,TTFunctionCall):
            return type_defining_node.is_intrinsic_call
        else:
            return False
   
    @property
    def is_elemental_function_call(self):
        type_defining_node = self.type_defining_node
        if isinstance(type_defining_node,TTFunctionCall):
            return type_defining_node.is_elemental_function_call
        else:
            return False

    @property
    def is_converter_call(self):
        type_defining_node = self.type_defining_node
        if isinstance(type_defining_node,TTFunctionCall):
            return type_defining_node.is_converter_call
        else:
            return False
 
    @property
    def result_depends_on_kind(self):
        type_defining_node = self.type_defining_node
        if isinstance(type_defining_node,TTFunctionCall):
            return type_defining_node.result_depends_on_kind
        else:
            return False

    def index_range_args(self):
        """:note: While multiple derived type member parts can have
        an argument list, only one can have index ranges as arguments.
        """
        rank_defining_node = self.rank_defining_node
        if isinstance(rank_defining_node,TTFunctionCall):
            yield from rank_defining_node.index_range_args()
        else:
            yield from ()

    def walk_values_ignore_args(self):
        """Only descend into elemental functions and yield their arguments,
        otherwise yield `self`."""
        if (self.is_function_call
           and self.is_elemental_function_call):
            for arg_name, arg_value in self.value.iter_actual_arguments():
                if not (
                    self.result_depends_on_kind 
                    and arg_name == "kind"
                  ):
                    yield from arg_value.walk_values_ignore_args()
        else:
            yield self

    def loop_bounds_as_str(self,
        only_consider_index_ranges=False,
        converter=traversals.make_cstr
      ):
        """:return: triple of lower bounds, upper bounds, and strides as str.
        :param bool only_consider_index_ranges: Only extract lower bounds, upper bounds
                                          and strides from TTIndexRange arguments.
        :param converter: Routine to convert the expressions to strings, 
                          defaults to `traversals.make_cstr`. Other legal option
                          is `traversals.make_fstr`.
        """
        assert isinstance(self.value,TTFunctionCall)
        if converter == traversals.make_cstr:
            var_name = self.cname
            bound_template = "{var}.{bound}({i})"
        else:
            assert converter == traversals.make_fstr
            bound_template = "{bound}({var},{i})"
        lbounds = []
        ubounds = []
        steps = []
        for i,ttarg in enumerate(self.value.args,1):
            if isinstance(ttarg,TTIndexRange):
                if ttarg.has_lbound():
                    lbounds.append(converter(ttarg.lbound))
                else:
                    lbounds.append(
                      bound_template.format(
                        bound="lbound",
                        var = var_name,
                        i = i
                      )
                    ) 
                if ttarg.has_ubound():
                    ubounds.append(converter(ttarg.ubound))
                else:
                    ubounds.append(
                      bound_template.format(
                        bound="ubound",
                        var = var_name,
                        i = i
                      ) 
                    )
                if ttarg.has_stride():
                    steps.append(converter(ttarg.stride))
                else:
                    steps.append(None)
            elif not only_consider_index_ranges:
                lbounds.append(converter(ttarg))
                ubounds.append(converter(ttarg))
                steps.append(None)
        return (lbounds, ubounds, steps)
    
    def fstr(self):
        return traversals.make_fstr(self.value)

    def overwrite_c_repr(self,
        name,
        args,
        use_square_brackets = False
      ):
        """Overwrites C representation of this TTValue as identifier plus optional
        list of arguments.
        :param str name: Overwrite name of the variable when generating C code.
        :param list args: List of argument expressions, may be empty, or None.
        :param bool use_square_brackets: Place argument list into C-style square brackets instead of parantheses.
        """
        self._c_repr_name = name
        self._c_repr_args = args
        self._c_repr_use_square_brackets = False 

    def cstr(self):
        if self._c_repr_name != None:
            result = self.cname
            if self._c_repr_args != None:
                arg_list = "{}".format(
                  ",".join([traversals.make_cstr(arg) for arg in self._c_repr_args])
                )
                if self._c_repr_use_square_brackets:
                    result += "[" + arg_list + "]"
                else:
                    result += "(" + arg_list + ")"
                return result
        else:
            return traversals.make_cstr(self.value)

class TTLvalue(TTValue):
    def __str__(self):
        return "TTLvalue(val:"+str(self.value)+")"
    __repr__ = __str__

class TTRvalue(TTValue):
    def __str__(self):
        return "TTRvalue(val:"+str(self.value)+")"
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
  
    @property
    def is_logical(self):
        return self == OperatorType.LOGIC
  
    @property
    def is_numeric(self):
        return not self.is_logical

class TTUnaryOp(ArithExprNode):
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

    @property
    def yields_array(self):
        return self.opd.yields_array
    @property
    def yields_scalar(self):
        return not self.yields_array
    @property
    def yields_contiguous_array(self):
        return self.opd.yields_contiguous_array
    @property
    def yields_full_array(self):
        return self.opd.yields_full_array

    def walk_values_ignore_args(self):
        yield from walk_values_ignore_args(self.opd)
 
    def _op_c_template(self):
        return TTUnaryOp.f2c.get(self.op.lower())
    def cstr(self):
        if _need_to_add_brackets(self,self.opd):
            return self._op_c_template().format(
              r="(" + self.opd.cstr() + ")"
            )
        else:
            return self._op_c_template().format(
              r=self.opd.cstr()
            )
    
    def fstr(self):
        if _need_to_add_brackets(self,self.opd):
            return self.op + "(" + self.opd.fstr() + ")"
        else:
            return self.op + self.opd.fstr()

class TTBinaryOpChain(ArithExprNode):
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
        yield from self.operands
    
    @property
    def yields_array(self):
        """:note: Assumes that semantics check has already performed.
        """
        for opd in self.operands:
            if opd.yields_array:
                return True
        return False
    
    @property
    def yields_scalar(self):
        """:note: Assumes that semantics check has already performed.
        """
        return not self.yields_array

    @property
    def yields_contiguous_array(self):
        """:return: If the operands are a mix of at least one contiguous array and scalars.
        :note: Assumes that semantics check has already performed, i.e.
               that all array ranks are consistent.
        :note: Full array expression is always contiguous too.
        """
        one_contiguous_array_found = False
        for opd in self.operands:
            yields_contiguous_array = opd.yields_contiguous_array
            if not (
              yields_contiguous_array
              or opd.is_scalar 
            ):
               return False
            if yields_contiguous_array:
                one_contiguous_array_found = True
        return one_contiguous_array_found
    
    @property
    def yields_full_array(self):
        """:return: If the operands are a mix of at least one full array and scalars.
        :note: Assumes that semantics check has already performed, i.e.
               that all array ranks are consistent.
        """
        one_full_array_found = False
        for opd in self.operands:
            yields_full_array = opd.yields_full_array
            if not (
              yields_full_array
              or opd.is_scalar 
            ):
               return False
            if yields_full_array:
                one_full_array_found = True
        return one_full_array_found
    
    def walk_values_ignore_args(self):
        for opd in self.operands:
            yield from opd.walk_values_ignore_args()
    
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
              op = op
            )
        return result

class TTArithExpr(ArithExprNode):

    def _assign_fields(self, tokens):
        self.expr = tokens[0] # either: rvalue,unary op,binary op

    def child_nodes(self):
        yield self.expr
    
    @property
    def type(self):
        return self.expr.type
    @property
    def rank(self):
        return self.expr.rank
    @property
    def bytes_per_element(self):
        return self.expr.bytes_per_element

    @property
    def yields_array(self):
        return self.expr.yields_array
    @property
    def yields_contiguous_array(self):
        return self.expr.yields_contiguous_array
    @property
    def yields_full_array(self):
        return self.expr.yields_full_array
    @property
    def yields_scalar(self):
        return self.expr.yields_scalar
    
    def walk_values_ignore_args(self):
        yield from self.expr.walk_values_ignore_args()

    @property
    def applies_transformational_functions_to_arrays(self):
        """:return: If the rank or shape of arrays in this expression
        might be changed due to function calls.
        :todo: How does this relate to 'elemental' property?
               Seems it is more careful than checking that
               every function is elemental in the expression.
        """
        for child in self.walk_preorder():
            if (isinstance(child,TTFunctionCall)
               and child.is_function_call
               and not child.is_elemental):
                for arg in child.args:
                    if not isinstance(arg,TTIndexRange) and arg.yields_array:
                        return True
        return False    

    @property
    def contains_array_access_with_index_array(self):
        """:return: If an array of indices is passed as index to 
        an array access expression.

        Examples:

        array([1,2,3])
        array1(array2)
        
        :todo: How does this relate to 'contiguous' property?
               If index argument is non-contiguous the result is non-contiguous.
               If index argument is contiguous the result is contiguous.
               Shall we just assume that index arrays always destroy contiguity.
        """
        for child in self.walk_preorder():
            if (isinstance(child,TTFunctionCall)
               and child.is_array):
                for arg in child.args:
                    # treat index_range without stride as contiguous array?
                    if not isinstance(arg,TTIndexRange) and arg.yields_array:
                        return True
        return False    

    @property
    def is_unprefixed_single_value(self):
        return isinstance(self.expr,TTValue)
    
    @property
    def is_prefixed_single_value(self):
        return (
                 isinstance(self.expr,TTUnaryOp)
                 and isinstance(self.expr.opd,TTValue)
               )

    def cstr(self):
        return self.expr.cstr()

    def fstr(self):
        return self.expr.fstr()


class TTKeywordArgument(ArithExprNode):
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
    @property
    def yields_array(self):
        return self.value.yields_array
    @property
    def yields_scalar(self):
        return self.value.is_scalar
    @property
    def yields_contiguous_array(self):
        return self.value.yields_contiguous_array
    @property
    def yields_full_array(self):
        return self.value.yields_full_array

    def child_nodes(self):
        yield self.value
    
    def walk_values_ignore_args(self):
        yield from self.value.walk_values_ignore_args()

    def cstr(self):
        return self.key + "=" + self.value.cstr() + ";\n"

    def fstr(self):
        return self.key + "=" + self.value.fstr()


class TTAssignment(base.TTStatement):

    def _assign_fields(self, tokens):
        self.lhs, self.rhs = tokens
 
    @property
    def type(self):
        return self.lhs.type
    @property
    def rank(self):
        return self.lhs.rank
    @property
    def bytes_per_element(self):
        return self.lhs.bytes_per_element
 
    @property
    def is_copy(self):
        """:return: if this assignment is a copy operation"""
        return (
          self.rhs.is_unprefixed_single_value
          and (
            (self.lhs.is_scalar
            and self.rhs.is_scalar)
            or
            (self.lhs.yields_contiguous_array
            and self.rhs.yields_contiguous_array)
          )
        )
    
    @property
    def is_setval(self):
        """:return: if this assignment is a copy operation"""
        return (
          self.lhs.yields_contiguous_array
          and self.rhs.is_unprefixed_single_value
          and self.rhs.is_scalar
        )
    
    def is_simple_array_intrinsic_call(self):
        """Checks if this is an expression of the form 
        <scalar> = [+-] <intrinsic>(<array|subarray>)
        where the intrinsic is one of minval,maxval,sum.
        No DIM or MASK argument is allowed.
        """
        if (self.lhs.is_scalar and self.rhs.is_scalar):
            if self.rhs.is_unprefixed_single_value:
                candidate = self.rhs.expr
            elif self.rhs.is_prefixed_single_value:
                candidate = self.rhs.expr.opd
            else:
                return False
            if (
                candidate.value.is_function_call 
                and candidate.value.is_intrinsic_call
              ):
                if (
                  candidate.symbol_info.name in ["minval","maxval","sum"]
                  and len(candidate.args) == 1
                  and candidate.is_argument_specified("array")
                ):
                  array_arg = candidate.get_argument("array")
                  return (
                    array_arg.is_unprefixed_single_value
                    and array_arg.expr.is_array
                  )
        return False 

    #todo: Detect (hip-)BLAS routines too?

    def walk_values_ignore_args(self):
        yield from self.lhs.walk_values_ignore_args()
        yield from self.rhs.walk_values_ignore_args()

    def child_nodes(self):
        yield self.lhs
        yield self.rhs
    def cstr(self):
        return self.lhs.cstr() + "=" + self.rhs.cstr() + ";\n"
    def fstr(self):
        return self.lhs.fstr() + "=" + self.rhs.fstr() + ";\n"

class TTSubroutineCall(base.TTStatement):

    def _assign_fields(self, tokens):
        self._subroutine = tokens[0]

    def cstr(self):
        return self._subroutine.cstr() + ";"

def set_arith_expr_parse_actions(grammar):
    grammar.logical.setParseAction(TTLogical)
    grammar.character.setParseAction(TTCharacter)
    grammar.number.setParseAction(TTNumber)
    grammar.identifier.setParseAction(TTIdentifier)
    grammar.rvalue.setParseAction(TTRvalue)
    grammar.lvalue.setParseAction(TTLvalue)
    grammar.derived_type_elem.setParseAction(TTDerivedTypePart)
    grammar.function_call.setParseAction(TTFunctionCall)
    grammar.index_range.setParseAction(TTIndexRange)
    grammar.arith_expr.setParseAction(TTArithExpr)
    grammar.complex_constructor.setParseAction(
        TTComplexConstructor)
    grammar.array_constructor.setParseAction(
        TTArrayConstructor)
    grammar.keyword_argument.setParseAction(TTKeywordArgument)
    # statements
    grammar.assignment.setParseAction(TTAssignment)
    grammar.fortran_subroutine_call.setParseAction(TTSubroutineCall)
