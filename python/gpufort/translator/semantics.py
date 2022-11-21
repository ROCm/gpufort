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
  
    def _check_function_call_arg_positions(self,ttnode,scope):
        """
        - check that keyword arguments do not before positional arguments
        - check that no argument is specified as positional argument
          and keyword argument
        - check if all required arguments have been specified 
        """
        found_named_arg = False
        dummy_arg_names = ttnode.dummy_args # expected arguments
        used_arg_names = []
        # check that keyword arguments do not before positional arguments
        # check that no argument is specified as positional argument
        # and keyword argument
        for i,arg in enumerate(ttnode.args):
            if isinstance(arg,tree.TTKeywordArgument):
                found_named_arg = True
                key = arg.key.lower()
                if key in used_arg_names:
                    raise util.error.SemanticError(
                      "named argument '{}' already present as positional or keyword argument".format(
                        key
                      )
                    )
                used_arg_names.append(key)
            else:
                if found_named_arg:
                    raise util.error.SemanticError("positional argument after named argument")
                used_arg_names.append(dummy_arg_names[i])
        # check if all required arguments have been specified 
        for dummy_arg_name in dummy_arg_names:
            if dummy_arg_name not in used_arg_names:
                if not ttnode.get_expected_argument_is_optional(dummy_arg_name):
                    raise util.error.SemanticError(
                      "required argument '{}' not specified".format(
                        dummy_arg_name
                      )
                    )
        return used_arg_names # assumed to be all in lower case
  
    def _check_function_call_ttarg_type(self,ttfunc,ttarg,scope):
        """Check that ttargument type and kind agree.
        """
        actual_type = ttarg.type
        expected_type = ttfunc.get_expected_ttargument_type(ttarg_name)
        if actual_type != expected_type:
            raise util.error.SemanticError(
               "function {} '{}': ttargument type mismatch; expected type '{}' instead of '{}'".format(
                 func_type, func_name, expected_type, actual_type
              )                         
            )                           
        actual_kind = ttarg.kind
        actual_bytes_per_element = ttarg.bytes_per_element
        expected_kind = ttfunc.get_expected_ttargument_kind(ttarg_name)
        expected_bytes_per_element = _lookup_bytes_per_element1(
          expected_kind
        )
        if actual_bytes_per_element != expected_bytes_per_element:
            raise util.error.SemanticError(
               "function {} '{}': ttargument kind mismatch; expected kind '{}' instead of '{}'".format(
                 func_bytes_per_element, func_name,
                 expected_kind, actual_kind
              )                         
            )                           

    def _check_elemental_function_call_args(self,ttfunc,used_arg_names,scope):
        """Rank of the elemental function result. All arguments must
        have the same rank
        - and same number of elements if this information can be deduced (not implemented)
        :param arg_names: Lower-case argument names.
        """
        max_rank = 0
        func_name = ttfunc.name
        func_name_lower = func_name.lower()
        func_type = "intrinsic" if ttfunc.is_intrinsic_call else "function"
        for arg_name in used_arg_names:
            ttarg = ttfunc.get_actual_argument(arg_name)
            # check rank
            if ttarg.rank > 0:
                if (ttfunc.result_depends_on_kind
                   and arg_name == "kind"):
                     raise util.error.SemanticError(
                       "elemental {} '{}': 'kind' argument must be scalar integer".format(
                         func_type, func_name
                       )
                     )
                elif max_rank > 0 and ttarg.rank != max_rank:
                    raise util.error.SemanticError(
                      "elemental {} '{}': incompatible argument ranks: '{}' vs '{}'".format(
                        func_type, func_name, ttarg.rank, max_rank
                      )
                    )
            max_rank = max(ttarg.rank,max_rank)
            # check type
            if ttfunc.is_converter_call:
                if arg_name != "kind":
                    if (func_name.lower() in ["real","int","cmplex"]
                       and ttarg.type not in ["integer","real","complex"]):
                        raise util.error.SemanticError(
                           "conversion {} '{}': expected numeric type argument".format(
                             func_type, func_name, ttarg.rank, max_rank
                          )
                        )
                    elif (func_name.lower() == "logical"
                         and ttarg.type != "logical"):
                        raise util.error.SemanticError(
                           "conversion {} '{}': expected numeric type argument".format(
                             func_type, func_name, ttarg.rank, max_rank
                          )
                        )
            else:
                self._check_function_call_arg_type(ttfunc,ttarg,scope)
                                                
    def _check_other_function_call_args(self,ttfunc,used_arg_names,scope):
        """Check that the arguments of non-elemental functions match."""
        for arg_name in used_arg_names:
            ttarg = ttfunc.get_actual_argument(arg_name)
            self._check_function_call_arg_type(ttfunc,ttarg,scope)
 
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
        #print(ttnode.symbol_info)
        if ttnode.is_array_expr:
            if ttnode.type != "type":
                self._lookup_bytes_per_element(ttnode)
            self._check_array_indices(ttnode,scope)
        else: # is function call
            used_arg_names = self._check_function_call_arg_positions(ttnode,scope)
            if ttnode.is_elemental_function_call:
                self._check_elemental_function_call_args(ttnode,used_arg_names,scope)
            else:
                self._check_other_function_call_args(node,used_arg_names,scope)
            if ttnode.type != "type":
                self._lookup_bytes_per_element(ttnode)
    
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
          ttnode.opd.type 
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
    
    def resolve_assignment(self,ttassignment,scope):
        """Resolve the types in an arithmetic expression parse tree.
        Post-order traversal is mandatory as this traverses descedants
        before their parents.
        :raise util.error.LookupError: if a symbol's type could not be determined.
        :note: Can also be applied to subtrees contained in TTArithExpr.
        """
        self.resolve_arith_expr(ttassignment.lhs,scope)
        self.resolve_arith_expr(ttassignment.rhs,scope)
        if ttassignment.lhs.rank != ttassignment.rhs.rank:
            raise util.error.SemanticError("rank mismatch between LHS and RHS")
        if ( ttassignment.lhs.type != ttassignment.rhs.type
             or ttassignment.lhs.bytes_per_element != ttassignment.rhs.bytes_per_element ):
            raise util.error.SemanticError("type mismatch between LHS and RHS")
