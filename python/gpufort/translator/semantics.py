# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import util
from gpufort import indexer

from . import tree
from . import opts

class Semantics:

    def __init__(self):
        self.allow_real_as_array_index = False
        self.allow_arrays_as_array_index = False
        self.warnings = []

    def _lookup_bytes_per_element1(self,ftype,kind):
        kind = kind.lower() if isinstance(kind,str) else None
        try:
            return opts.fortran_type_2_bytes_map[
              ftype.lower()][kind]
        except KeyError as ke:
            raise util.error.LookupError(
              ("no number of bytes found for type '{}' and kind '{}' in"
              + "'translator.opts.fortran_type_2_bytes_map'").format(ftype,kind)
            ) from ke
    
    def _lookup_bytes_per_element(self,ttnode):
        ttnode.bytes_per_element = self._lookup_bytes_per_element1(
          ttnode.type, ttnode.kind
        )
    
    def _resolve_literal(self,ttnode,scope):
        if isinstance(ttnode,(tree.TTLogical,
                              tree.TTNumber)):
            self._lookup_bytes_per_element(ttnode)
    
    def _resolve_identifier(self,ttnode,scope):
        ttnode.symbol_info = indexer.scope.search_scope_for_var(scope,ttnode.fstr())
        self._lookup_bytes_per_element(ttnode)

    def _check_array_indices(self,ttnode,scope):
        # rank of the array section expression != array rank
        # Hence we cannot use ttnode.rank
        array_rank = ttnode.symbol_info["rank"]
        if len(ttnode.args) != array_rank:
            raise util.error.SemanticError(
              "{actual} instead of expected {expected} index "+
              "expressions passed to array variable '{var}'.".format(
                actual=len(ttnode.args),
                expected=array_rank,
                var=ttnode.name 
              )
            )
        for arg in ttnode.args:
            if not isinstance(arg,tree.TTSlice):
                if arg.type == "integer":
                    if not self.allow_arrays_as_array_index and arg.rank > 0:
                        raise util.error.SemanticError("array index is array")
                elif ttnode.type == "real":
                    if self.allow_real_as_array_index:
                        self.warnings.append("legacy extension: array index is real")
                    else:
                        raise util.error.SemanticError("array index is real")
                else:
                    raise util.error.SemanticError(
                      "index must be either of integer "
                      + "or real type (legacy extension)"
                    )
  
    def _check_function_call_arguments(self,ttnode,scope):
        found_named_arg = False
        for arg in ttnode.args:
            if isinstance(arg,tree.TTKeywordArgument):
                found_named_arg = True
            else:
                if found_named_arg:
                    raise util.error.SemanticError("positional argument after named argument")
 
    def _resolve_function_call(self,ttnode,scope):
        """
        - resolve type of the expression: function call vs array section
        - in case of function call,
          - check that number of args is equal to number of dummy args
          - check that keyword arguments appear only after positional argument
            - next level: check type of arguments, compare to that of function argument list
          - check if elemental or transformational intrinsic
          - handle conversion functions
        - in case of array section:
          - check that number of args is equal to rank
          - check that arguments are of integer type
        """
        ttnode.symbol_info = indexer.scope.search_scope_for_value_expr(
          scope,
          # todo: redundant for vars, should just use the name of the function
          # instead of fstr
          indexer.scope.create_index_search_tag(ttnode.fstr())
        )
        print(ttnode.symbol_info)
        if ttnode.type != "type":
            self._lookup_bytes_per_element(ttnode)
        if ttnode.is_array_expr():
            self._check_array_indices(ttnode,scope)
        else: # is function call
            self._check_function_call_arguments(ttnode,scope)
    
    def _resolve_derived_type(self,ttnode,scope):
        """ - check that not more than one part reference has nonzero rank, e.g.
              disallows expressions such as "a(:)%b(:)
            - check that element is actually member of the value type
        """
        already_found_part_reference_with_nonzero_rank = False
        prefix = ""
        def lookup_symbol_info_(type_or_element): 
            nonlocal already_found_part_reference_with_nonzero_rank
            nonlocal prefix
            type_or_element.symbol_info =\
              indexer.scope.search_scope_for_value_expr(
                scope,
                indexer.scope.create_index_search_tag(
                  prefix + type_or_element.fstr()
                )
              )
            if type_or_element.rank > 0: 
               if already_found_part_reference_with_nonzero_rank:
                   raise error.SemanticError(
                     "Two or more part references with nonzero rank must not be specified"
                   )
               already_found_part_reference_with_nonzero_rank = True

        # traverse: a%b%c%d -> a%(b%(c%d))
        for ttpart in ttnode.walk_derived_type_parts_preorder():
            lookup_symbol_info_(ttpart.derived_type)
            prefix += ttpart.derived_type.fstr() + "%"
            if not isinstance(ttpart.element,tree.TTDerivedTypePart):
                lookup_symbol_info_(ttpart.element)
                if ttpart.element.type != "type":
                    self._lookup_bytes_per_element(ttpart.element)
    
    def _check_complex_arith_expr(self,ttcomplexarithexpr,scope):
        """Check if a complex type is constructed of compatible components:
        - real and integer are allowed
        - all components must have rank 0
        """ 
        for child in self.child_nodes:
            if child.type not in [tree.Literal.REAL,
                                  tree.Literal.INTEGER]:
                raise error.SemanticError("components of complex must be real or integer")
            elif child.rank != 0:
                raise error.SemanticError("rank of complex components must be 0")
    
    def _compare_op_and_opd_type(self,op_type,opd_type):
        """check if operator and operand type agree."""
        if opd_type in ["real","integer"]:
            if op_type == tree.OperatorType.LOGIC:
                raise util.error.SemanticError("cannot apply logical binary operator to real type")
        elif opd_type == "logical":
            if op_type != tree.OperatorType.LOGIC:
                raise util.error.SemanticError("cannot apply numeric binary operator to logical type")
        else:
            raise util.error.SemanticError(
              "type '{}' not allowed in arithmetic expression".format(opd_type)
            )
    
    def _check_unary_op(self,ttnode,scope):
        self._compare_op_and_opd_type(
          ttnode.operator_type,
          opd_type
        )
    
    def _resolve_binary_op(self,ttnode,scope):
        """
        :note: Assumes that all operands have already been resolved."""
        op_type = ttnode.operator_type
        # first operand
        first_opd = ttnode.operands[0]
        current_type = first_opd.type
        current_rank = first_opd.rank
        current_bytes_per_element = first_opd.bytes_per_element
        self._compare_op_and_opd_type(op_type,first_opd.type)
        # reamining operands
        for opd in ttnode.operands[1:]:
            self._compare_op_and_opd_type(op_type,opd.type)
            # rank, can apply binary op to scalar-array pair
            # or pair of arrays with same rank
            if current_rank > 0:
                if opd.rank > current_rank:
                    raise util.error.SemanticError(
                      "inconsistent rank of operands, {} vs. {}".format(
                        current_rank, opd.rank
                      )
                    )
            else:
                current_rank = opd.rank
            # bytes per element
            if current_type == "real":
               if opd.type == "real":
                    current_bytes_per_element = max(
                      current_bytes_per_element,
                      opd.bytes_per_element
                    )
            elif current_type == "integer":
                if opd.type == "integer":
                     current_bytes_per_element = max(
                       current_bytes_per_element,
                       opd.bytes_per_element
                     )
                elif opd.type == "real":
                    current_type = "real" 
                    current_bytes_per_element = opd.bytes_per_element
            elif current_type == "logical":
                current_bytes_per_element = max(
                  current_bytes_per_element,
                  opd.bytes_per_element
                )
        ttnode.rank = current_rank
        ttnode.type = current_type
        ttnode.bytes_per_element = current_bytes_per_element
   
    def _resolve_value(self,ttnode,scope):
        if isinstance(ttnode._value,tree.Literal):
            self._resolve_literal(ttnode._value,scope)
        elif isinstance(ttnode._value,tree.TTIdentifier):
            self._resolve_identifier(ttnode._value,scope)
        elif isinstance(ttnode._value,tree.TTFunctionCall):
            self._resolve_function_call(ttnode._value,scope)
        elif isinstance(ttnode._value,tree.TTDerivedTypePart):
            self._resolve_derived_type(ttnode._value,scope)
 
    def resolve_arith_expr(self,ttarithexpr,scope):
        """Resolve the types in an arithmetic expression parse tree.
        Post-order traversal is mandatory as this traverses descedants
        before their parents.
        :raise util.error.LookupError: if a symbol's type could not be determined.
        :note: Can also be applied to subtrees contained in TTArithExpr.
        """
        assert isinstance(ttarithexpr,tree.TTNode), type(ttarithexpr)
        for ttnode in ttarithexpr.walk_postorder():
            if isinstance(ttnode,tree.TTValue):
                self._resolve_value(ttnode,scope)
            elif isinstance(ttnode,tree.TTUnaryOp):
                self._check_unary_op(ttnode,scope)
            elif isinstance(ttnode,tree.TTBinaryOpChain):
                self._resolve_binary_op(ttnode,scope)
