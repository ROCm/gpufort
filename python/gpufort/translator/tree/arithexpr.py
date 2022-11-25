# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#from translator_base import *
"""

Generators:

* walk_values_ignore_args: Certain nodes are equipped with this generator.
  This generator can be used to analyze the operands of 
  Fortran array expressions. Unlike the `walk_preorder` and `walk_postorder`
  routines, the generator does not descend into function call arguments and array indices.
  The only exception being elemental function calls.
  Here, the generator descends directly into the argument list and does
  not yield the elemental function call expression itself.
  It further does not yield any unary and binary operators. 
  It exclusively yields TTValue instances.
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

class Literal(base.TTNode):
    LOGICAL = "logical"
    CHARACTER = "character"
    INTEGER = "integer"
    REAL = "real"
    COMPLEX = "complex"
            
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
    def min_rank(self): return 0
    @property
    def is_scalar(self): return True
    @property
    def is_array(self): return False
    @property
    def is_contiguous_array(self): return False
    @property
    def is_full_array(self): return False

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

    @property 
    def rank(self):
        assert self.partially_resolved
        return self.symbol_info.rank

    @property
    def min_rank(self):
        """:return: Minimum rank if an '*' or '..' is present in the array bounds
        specification. Otherwise, returns the same value as `rank`.
        :note: This routine only makes sense in the context of procedure arguments.
        """
        rank = self.rank
        if rank == indexer.indexertypes.IndexVariable.ANY_RANK_GREATER_THAN_MIN_RANK:
            return len([e for e in bounds if "*" in bounds])
        elif rank == indexer.indexertypes.IndexVariable.ANY_RANK:
            return 0
        else:
            return rank
    
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
  
    def fstr(self):
        return self.name

    def cstr(self):
        return self.name.lower()

    def __str__(self):    
        return "TTIdentifier(name:"+str(self._name)+")"
    __repr__ = __str__


class TTFunctionCall(VarExpr):
 
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
        """
        Checks for `kind` argument and returns expression.
        If `kind` argument not found, returns `None` (default kind).
        :note: Assumes semantics have been analyzed before.
        """
        if ( self.is_function_call 
             and self.result_depends_on_kind ):
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
            return len([arg for arg in self.slice_args()])
        else: 
            return self._get_type_defining_record().rank

    @property
    def min_rank(self):
        return self.rank

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

    @property
    def is_array_expr(self):
        """:return: if this expression represents
        an array section or array element (array section of size 1)."""
        return isinstance(self.symbol_info,indexer.indexertypes.IndexVariable)
   
    @property
    def is_array(self):
        """If this an array expression, check if there is an index range.
        If this is an elemental function, check if one argument
        is an array.
        """
        # todo incomplete
        if ( self.is_function_call
             and self.is_elemental_function_call ):
            for arg in self.args:
                if arg.is_array:
                    return True
        elif self.is_array_expr:
            for arg in self.args:
                if isinstance(arg,TTSlice):
                    return True
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
        if ( self.is_function_call
             and self.is_elemental_function_call ):
            one_array_is_contiguous = False
            for name, value in self.iter_actual_arguments():
                if value.is_contiguous_array:
                    one_array_is_contiguous = True
                if not (
                    value.is_contiguous_array
                    or value.is_scalar
                  ):
                    if self.result_depends_on_kind:
                        if name != "kind":
                            return False
                    else:
                        return False
            return one_array_is_contiguous
        elif self.is_array_expr:
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
        else:
            return False
    
    @property
    def is_scalar(self):
        """Checks if none of the array index arguments 
        are index ranges.
        """
        return not self.is_array

    @property
    def is_full_array(self):
        """Checks if this array section expression
        covers all elements of an original array.
        For elemental functions
        """
        if ( self.is_function_call
             and self.is_elemental_function_call ):
            one_array_is_full = False
            for name, value in self.iter_actual_arguments():
                if value.is_full_array:
                    one_array_is_full = True
                if not (
                    value.is_full_array
                    or value.is_scalar
                  ):
                    if self.result_depends_on_kind:
                        if name != "kind":
                            return False
                    else:
                        return False
            return one_array_is_full
        elif self.is_array_expr:
            for arg in self.args:
                if (not isinstance(arg,TTSlice)
                   or arg.has_lbound()
                   or arg.has_ubound()
                   or arg.has_stride()):
                    return False
            return True
        else:
            return False

    @property
    def is_function_call(self):
        return not self.is_array_expr
    
    @property
    def is_intrinsic_call(self):
        assert self.is_function_call
        return self.symbol_info.is_intrinsic
   
    @property
    def is_elemental_function_call(self):
        assert self.is_function_call
        return self.symbol_info.is_elemental

    @property
    def is_converter_call(self):
        """:note: Conversions are always elemental."""
        assert self.is_elemental_function_call
        return self.symbol_info.is_conversion 
    
    @property
    def result_depends_on_kind(self):
        """:note: Conversions are always elemental."""
        assert self.is_elemental_function_call
        return self.symbol_info.result_depends_on_kind

    @property
    def dummy_args(self):
        """Names of arguments in their expected order
        when all arguments are supplied as positional arguments.
        """
        assert self.is_function_call
        return self.symbol_info["dummy_args"] 

    # todo: this should better go into a symbol_info
    # that indexer.scope creates when you look up a variable
    def get_expected_argument(self,arg_name):
        assert self.is_function_call
        return self.symbol_info.get_argument(arg_name)

    # todo: this should better go into a symbol_info
    # that indexer.scope creates when you look up a variable
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

class TTSlice(base.TTNode):

    def _assign_fields(self, tokens):
        self.lbound, self.ubound, self.stride =\
          None, None, None 
        if len(tokens) == 1:
            self.ubound = tokens[0]
        elif len(tokens) == 2:
            self.lbound = tokens[0]
            self.ubound = tokens[1]
        elif len(tokens) == 3:
            self.lbound = tokens[0]
            self.ubound = tokens[1]
            self.stride = tokens[2]
        self._cstr = None

    @property
    def type(self):
        return self.ubound.type
    @property
    def kind(self):
        return self.ubound.kind
    @property
    def bytes_per_element(self):
        return self.ubound.bytes_per_element
    @property
    def rank(self):
        return self.ubound.rank

    def has_lbound(self):
        return self.lbound != None
    def has_ubound(self):
        return self.ubound != None
    def has_stride(self):
        return self.stride != None

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

    def overwrite_cstr(self,cstr):
        self._cstr = cstr

    def cstr(self):
        #return self.size(traversals.make_cstr)
        if self._cstr == None:
            return "/*TODO fix this BEGIN*/{0}/*fix END*/".format(self.fstr())
        else:
            return self._cstr 
    
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

class TTDerivedTypePart(base.TTNode):

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

class TTComplexConstructor(base.TTNode):

    def _assign_fields(self, tokens):
        self.real, self.imag = tokens[0]

    def child_nodes(self):
        yield self.real; yield self.imag

    @property
    def type(self):
        return Literal.COMPLEX

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
    def min_rank(self):
        return 0
    @property
    def is_scalar(self): return True
    @property
    def is_array(self): return False
    @property
    def is_contiguous_array(self): return False
    @property
    def is_full_array(self): return False
    
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
                real=traversals.make_cstr(self.real),\
                imag=traversals.make_cstr(self.imag))

    def fstr(self):
        return "({real},{imag})".format(\
                real=traversals.make_fstr(self.real),\
                imag=traversals.make_fstr(self.imag))

class TTArrayConstructor(base.TTNode):

    def _assign_fields(self, tokens):
        self.entries = tokens

    def child_nodes(self):
        yield from self.entries

    @property
    def type(self):
        return self.entries[0].type
    @property
    def bytes_per_element(self):
        return self.entries[0].bytes_per_element
    @property
    def rank(self):
        return 1
    @property
    def min_rank(self):
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
    
    def walk_values_ignore_args(self):
        yield self

    def cstr(self):
        # todo array types need
        assert False, "Not implemented"

    def fstr(self):
        return "[{}]".format(
          ",".join([traversals.make_fstr(e) for e in self.entries])
        )

class TTValue(base.TTNode):
    
    def _assign_fields(self, tokens):
        self.value = tokens[0]
        # for overwriting C representation
        self._cname = None 
        self._cargs = []

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
    
    @property
    def min_rank(self):
        return self.rank_defining_node.min_rank

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
          (TTIdentifier,TTFunctionCall,TTDerivedTypePart))
        return self.value.name
   
    @property
    def cname(self):
        """C-style identifier of the expression, i.e. '%' replaced by '.'.
        :note: May still contain index range expressions.
        :note: If self._cname is set, it is returned instead.
        """
        if self._cname != None:
            return self._cname
        else:
            return self.name.replace("%",".") 
 
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

    def slice_args(self):
        """:note: While multiple derived type member parts can have
        an argument list, only one can have index ranges as arguments.
        """
        rank_defining_node = self.rank_defining_node
        if isinstance(rank_defining_node,TTFunctionCall):
            yield from rank_defining_node.slice_args()
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
    
    def fstr(self):
        return traversals.make_fstr(self.value)

    def overwrite_cstr(cname,cargs):
        """:param str cname: Overwrite name of variable when generating C expression.
        :param list(str) cargs: Overwrite argument list when generating C expression.
        :note: Should only be applied to derived type member access expressions
        if the rank and type defining node coincide, i.e. none or only the innermost element
        is an subarray or array.
        """
        self._cname = cname
        self._cargs = cargs

    def cstr(self):
        if self._cname != None:
            result = self.cname
            if len(self._cargs):
                result += "({})".format(
                  traversal.make_cstr(self._cargs)
                )
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
    
    @property
    def min_rank(self):
        return self.opd.min_rank

    @property
    def is_array(self):
        return self.opd.is_array
    
    @property
    def is_scalar(self):
        return not self.is_array
        
    @property
    def is_contiguous_array(self):
        return self.opd.is_contiguous_array
    
    @property
    def is_full_array(self):
        return self.opd.is_full_array

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
    def min_rank(self):
        return self.rank 
 
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
    
    @property
    def is_array(self):
        """:note: Assumes that semantics check has already performed.
        """
        for opd in self.operands:
            if opd.is_array:
                return True
        return False
    
    @property
    def is_scalar(self):
        """:note: Assumes that semantics check has already performed.
        """
        return not self.is_array

    @property
    def is_contiguous_array(self):
        """:return: If the operands are a mix of at least one contiguous array and scalars.
        :note: Assumes that semantics check has already performed, i.e.
               that all array ranks are consistent.
        :note: Full array expression is always contiguous too.
        """
        one_contiguous_array_found = False
        for opd in self.operands:
            is_contiguous_array = opd.is_contiguous_array
            if not (
              is_contiguous_array
              or opd.is_scalar 
            ):
               return False
            if is_contiguous_array:
                one_contiguous_array_found = True
        return one_contiguous_array_found
    
    @property
    def is_full_array(self):
        """:return: If the operands are a mix of at least one full array and scalars.
        :note: Assumes that semantics check has already performed, i.e.
               that all array ranks are consistent.
        """
        one_full_array_found = False
        for opd in self.operands:
            is_full_array = opd.is_full_array
            if not (
              is_full_array
              or opd.is_scalar 
            ):
               return False
            if is_full_array:
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

class TTArithExpr(base.TTNode):

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
    def min_rank(self):
        return self.expr.min_rank
    @property
    def bytes_per_element(self):
        return self.expr.bytes_per_element
    @property
    def is_array(self):
        return self.expr.is_array
    @property
    def is_contiguous_array(self):
        return self.expr.is_contiguous_array
    @property
    def is_full_array(self):
        return self.expr.is_full_array
    @property
    def is_scalar(self):
        return self.expr.is_scalar
    
    def walk_values_ignore_args(self):
        yield from self.expr.walk_values_ignore_args()

    def cstr(self):
        return self.expr.cstr()

    def fstr(self):
        return self.expr.fstr()


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
    @property
    def min_rank(self):
        return self.value.min_rank
    @property
    def is_array(self):
        return self.value.is_array
    @property
    def is_scalar(self):
        return self.value.is_scalar
    @property
    def is_contiguous_array(self):
        return self.value.is_contiguous_array
    @property
    def is_full_array(self):
        return self.value.is_full_array

    def child_nodes(self):
        yield self.value
    
    def walk_values_ignore_args(self):
        yield from self.value.walk_values_ignore_args()

    def cstr(self):
        return self.key + "=" + self.value.cstr() + ";\n"

    def fstr(self):
        return self.key + "=" + self.value.fstr()


class TTAssignment(base.TTNode):

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
    
    def walk_values_ignore_args(self):
        yield from self.lhs.walk_values_ignore_args()
        yield from self.rhs.walk_values_ignore_args()

    def child_nodes(self):
        yield self.rhs; yield self.rhs
    def cstr(self):
        return self.lhs.cstr() + "=" + self.rhs.cstr() + ";\n"
    def fstr(self):
        return self.lhs.fstr() + "=" + self.rhs.fstr() + ";\n"

class TTSubroutineCall(base.TTNode):

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
    grammar.tensor_slice.setParseAction(TTSlice)
    grammar.arith_expr.setParseAction(TTArithExpr)
    grammar.complex_constructor.setParseAction(
        TTComplexConstructor)
    grammar.array_constructor.setParseAction(
        TTArrayConstructor)
    grammar.keyword_argument.setParseAction(TTKeywordArgument)
    # statements
    grammar.assignment.setParseAction(TTAssignment)
    grammar.fortran_subroutine_call.setParseAction(TTSubroutineCall)
