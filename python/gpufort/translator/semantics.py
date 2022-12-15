# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import textwrap

from gpufort import util
from gpufort import indexer

from . import tree
from . import opts

class Semantics:
    """Class for performing semantics checks.
   
    Configurable class members:
    
    * **allow_array_index_to_be_array**:
      Allow array indices to be arrays itself, defaults to False.
    * **allow_array_index_to_be_real**:
      Allow array indices to be real (legacy extension), defaults to False.
    * **allow_interface_calls = False**:
      Allow calls to Fortran interfaces, defaults to False.
    * **acc_supported_device_types**:
      Supported ACC device types, defaults to ["host","nohost","nvidia","radeon","hip]
    * **acc_allow_unchecked_mapping_of_subarrays**:
      Supported ACC device types, defaults to ["host","nohost","nvidia","radeon","hip]
    
    Concepts:

    * Uses the walk_postorder method from the tree.TTNode class
      to ensure that child nodes are resolved before their
      parents, i.e. that the parse tree tree is traversed in 
      a bottom-up fashion.

    Todos:

    * Support interfaces. In this case, we need to filter
      candidate out of a number of available candidates that
      agrees with the passed arguments.
    """

    def __init__(self):
        """Constructor.

        :param scope: The scope to use for looking up variables and procedures.
        """
        self.allow_array_index_to_be_array = False
        self.allow_array_index_to_be_real = False
        self.allow_interface_calls = False
        self.acc_allow_unchecked_mapping_of_subarrays = True
        self.acc_supported_device_types = ["host","nohost","nvidia","radeon","hip"]
        #
        self.warnings = []
  
    def _lookup_bytes_per_element1(self,ftype,kind):
        """Lookup number of bytes required to store one instance of the type. 
        :note: As Fortran allows user-defined kinds, we use
               bytes per element to uniquely determine a datatype.
        :note: argument 'kind' might be not be lower case if the property 
               was derived from a literal such as '1.0_DP'.
        :note: We assume the Fortran type ('ftype') is already lower
               case as it stems from the symbol table.
        """
        kind = kind.lower() if kind != None else None
        try:
            return opts.fortran_type_2_bytes_map[ftype][kind]
        except KeyError as ke:
            if ftype == "type":
                return None
            else:
                raise util.error.LookupError("could not find bytes per element for"
                  + " type '{}' and kind '{}' in translator.opts.fortran_type_2_bytes_map".format(
                    ftype,kind
                  )
                ) from ke
    
    def _lookup_bytes_per_element(self,ttnode):
        """Shortcut for nodes with (resolved) kind attribute."""
        return self._lookup_bytes_per_element1(ttnode.type,ttnode.kind)

    def _default_integer_bytes_per_element(self):
        return self._lookup_bytes_per_element1(tree.FortranType.INTEGER,None)
        
    def _check_yields_expected_rank(self,ttnode,expected_rank,errmsg):
        """Checks if the expression has/yields the expected rank."""
        if ( expected_rank != ttnode.rank ):
            raise util.error.SemanticError(errmsg) 

    def _yields_expected_type(self,ttnode,expected_type,expected_kind):
        """:return: If the arithmetic expression node yields the expected type."""
        actual_type = ttnode.type
        #actual_kind = ttnode.kind
        #actual_bytes = self._lookup_bytes_per_element1(actual_type,actual_kind)
        actual_bytes = ttnode.bytes_per_element 
        expected_bytes = self._lookup_bytes_per_element1(expected_type,expected_kind)
        return (actual_type == expected_type 
               or actual_bytes == expected_bytes)

    def _yields_default_integer_type(self,ttnode):
        """:return: if the arithmetic expression node yields the default integer type."""
        return self._yields_expected_type(
          ttnode,errmsg,tree.FortranType.INTEGER,None
        )

    def _yields_default_real_type(self,ttnode):
        return self._yields_expected_type(
          ttnode,errmsg,tree.FortranType.REAL,None
        )

    def _yields_default_logical_type(self,ttnode):
        return self._yields_expected_type(
          ttnode,errmsg,tree.FortranType.LOGICAL,None
        )

    def _check_yields_expected_type(self,ttnode,errmsg,expected_type,expected_kind):
        """Checks if the expression has/yields the expected type (and kind).
        :raise util.error.SemanticError: If a type or kind mismatch was detected.
        """
        if not self._yields_expected_type(ttnode,expected_type,expected_kind):
            raise util.error.SemanticError(errmsg) 
    
    def _check_yields_integer_type_of_kind(self,ttnode,errmsg,expected_kind):
        self._check_yields_expected_type(
          ttnode,errmsg,tree.FortranType.INTEGER,expected_kind
        )
    
    def _check_yields_default_integer_type(self,ttnode,errmsg):
        self._check_yields_integer_type_of_kind(ttnode,errmsg,None)
    
    def _check_yields_real_type_of_kind(self,ttnode,errmsg,expected_kind):
        self._check_yields_expected_type(
          ttnode,errmsg,tree.FortranType.REAL,expected_kind
        )
    
    def _check_yields_default_real_type(self,ttnode,errmsg):
        self._check_yields_real_type_of_kind(ttnode,errmsg,None)
   
    def _check_yields_acc_handle_type(self,ttnode,errmsg):
        self._check_yields_integer_type_of_kind(ttnode,errmsg,"acc_handle_kind")
    
    def _check_yields_default_logical_type(self,ttnode,errmsg):
        self._check_yields_expected_type(
          ttnode,errmsg,tree.FortranType.LOGICAL,expected_kind
        )

    def _resolve_literal(self,ttnode,scope):
        """Looks up the bytes per element for the literal type."""
        ttnode.bytes_per_element = self._lookup_bytes_per_element(ttnode)
    
    def _check_array_constructor(self,ttnode,scope):
        """Checks that the elements all yield the same numeric, logical,
        or character type. Unlike in case of the complex constructor,
        different numeric (and other) types and kinds, 
        e.g. REAL(4) and REAL(8), cannot be mixed.
        :raise util.error.SemanticError: If a component has a different type than the first one.
        """
        errmsg = "matrix constructor: type mismatch, {} with {} bytes vs. {} with {} bytes"
        first_entry_type = ttnode.entries[0].type
        first_entry_bytes_per_element = ttnode.entries[0].bytes_per_element
        for entry in ttnode.entries[1:]:
            if (entry.type != first_entry_type
               or entry.bytes_per_element != first_entry_bytes_per_element):
                raise util.error.SyntaxError(errmsg.format(
                  first_entry_type,first_entry_bytes_per_element,
                  entry.type,entry.bytes_per_element 
                ))
    
    def _check_complex_constructor(self,ttnode,scope):
        """Checks that the components yield numeric types
        and derives the bytes per element of the complex-yielding
        expression.
        :raise util.error.SemanticError: If a component is not of integer or real type.
        """
        errmsg = "complex constructor components must be integer or real"
        current_type = ttnode.real.type
        current_bytes_per_element = ttnode.real.bytes_per_element
        for component in ttnode.child_nodes(): # ttnode.real, ttnode.imag
            if (not component.yields_numeric_type
               or component.yields_complex):
                raise util.error.SyntaxError(errmsg)
    
    def _resolve_identifier(self,ttnode,scope,name=None):
        if name == None:
            name = ttnode.name
        ttnode.symbol_info = indexer.scope.search_scope_for_var(scope,name)
        ttnode.bytes_per_element = self._lookup_bytes_per_element(ttnode)
 
    def _compare_operand_and_operator(self,ttop,ttopd):
        """Checks if operand and operator type agree.
        I.e. numeric operators cannot be applied to logical and character
        operands while logical operators cannot be applied to character and 
        numeric types.
        :param ttop: TTUnaryOp or TTBinaryOpChain
        :param ttopd: An arithmetic expression constitutent.
        :raise util.error.SemanticError: If the operator is not applicable to the operand.
        """  
        if ttop.operator_type.is_logical and not ttopd.yields_logical:
            raise util.error.SemanticError(
              "cannot apply logical operator '{}' on operand of type '{}'".format(
                ttop.op, ttopd.type
              )
            )
        elif ttop.operator_type.is_numeric and not ttopd.yields_numeric_type:
            raise util.error.SemanticError(
              "cannot apply numeric operator '{}' on operand of type '{}'".format(
                ttop.op, ttopd.type
              )
            )
 
    def _compare_type(self,
          this_type,
          this_bytes_per_element,
          other_type,
          other_bytes_per_element,
      ):
        """Type casting rules.
        :note: It must be ensured that the types are compatible
        before calling this routine.
        """
        if this_type == tree.FortranType.COMPLEX:
            if other_type == tree.FortranType.COMPLEX:
                this_bytes_per_element = max(
                  this_bytes_per_element,other_bytes_per_element
                ) # else
        elif this_type == tree.FortranType.REAL:
            if other_type == tree.FortranType.COMPLEX:
                this_type = other_type
                this_bytes_per_element = other_bytes_per_element
            elif other_type == tree.FortranType.REAL:
                this_bytes_per_element = max(
                  this_bytes_per_element,other_bytes_per_element
                ) # else
        elif this_type == tree.FortranType.INTEGER:
            if other_type == tree.FortranType.COMPLEX:
                this_type = other_type
                this_bytes_per_element = other_bytes_per_element
            elif other_type == tree.FortranType.REAL:
                this_type = other_type
                this_bytes_per_element = other_bytes_per_element
            elif other_type == tree.FortranType.INTEGER:
                this_bytes_per_element = max(
                  this_bytes_per_element,other_bytes_per_element
                ) # else
        elif other_type == tree.FortranType.LOGICAL:
            this_bytes_per_element = max(
              this_bytes_per_element,other_bytes_per_element
            ) # else
        return this_type, this_bytes_per_element 
    
    def _compare_binary_op_operand_ranks(self,
        ttopds,
      ):
        """Checks if the ranks of the operands agree, i.e. 
        if all operands are of the same rank or scalars.
        :return: Maximum rank of the operands.
        :see: _compare_operand_and_operator
        :raise util.error.SemanticError: If at least two operands are arrays with differing rank.
        """
        max_rank = ttopds[0].rank
        for opd in ttopds[1:]:
            other_rank = opd.rank
            if max_rank == 0 or other_rank == 0:
                max_rank = max(max_rank,other_rank)
            elif max_rank != other_rank:
                raise util.error.SemanticError(
                  "operands have incompatible rank, rank {} vs. rank {}".format(
                    max_rank,other_rank
                  )
                )
        return max_rank

    def _check_array_index(self,ttarg):
        """Checks if the array index is an integer -- or real
        if legacy extensions are enabled. Issues warnings
        if the index is not of default integer type.
        :raise util.error.SemanticError: If the type is not an integer -- or real
                                         if legacy extensions are enabled."""
        if ttarg.yields_real and self.allow_array_index_to_be_real:
            self.warnings.append("legacy extension: array index is of type real")
        elif ttarg.yields_integer:
            if ttarg.bytes_per_element != self._default_integer_bytes_per_element():
                self.warnings.append("array index is not default integer")
        else:
            raise util.error.SemanticError("array index must be integer")

    def _resolve_index_range(self,ttnode):
        """Check the type of index range components (lbound, ubound, stride),
        if specified, and derive the type and kind (bytes per element) of the
        overall expression.
        """
        for component in ttnode.child_nodes():
            self._check_array_index(component)
 
    def _check_array_indices(self,ttnode):
        """Checks that all indices are integer (or real if legacy extensions are enabled).
        Check that the rank of array indices is 1.
        :note: Accepts real indices if self.allow_array_index_to_be_real is set to True.
        :note: Accepts rank 1 array indices if the self.allow_array_index_to_be_real is set to True.
        :raise util.error.SemanticError:
        """
        for ttarg in ttnode.args:
            if isinstance(ttarg,tree.TTKeywordArgument):
                raise util.error.SyntaxError(
                  "illegal keyword argument '{}' in argument list of array".format(
                    ttarg.key
                  )
                )
            elif not isinstance(ttarg,tree.TTIndexRange):
                 self._check_array_index(ttarg)
                 if ttarg.rank > 0:
                     if not self.allow_array_index_to_be_array:
                         raise util.error.LimitationError(
                           "array index is array itself"
                         )
                     elif ttarg.rank > 1:
                         raise util.error.SemanticError(
                           "array index is an array with rank {}".format(
                             ttarg.rank
                           )
                         )
    
    def _compare_actual_and_expected_argument(self,
        ttfunccall,ttarg,expected_arg,
        check_rank = False,
        check_bytes_per_element = False
      ):
        """Compare actual and expected argument's type, bytes per element and rank.
        :param bool check_rank: Ensure expected rank and argument rank match.
        :param bool check_bytes_per_element: Ensure expected bytes per element 
                                             and actual bytes per element match.  
        """
        if not expected_arg.matches_type(ttarg.type):
            print(ttarg.type)
            if not expected_arg.matches_any_type:
                raise util.error.SemanticError(
                  "{}: argument '{}' has wrong type, expected {}".format(
                    ttfunccall.name,expected_arg.name,
                    expected_arg.full_type_as_str
                  )
                )
        # rank
        if check_rank:
            if not expected_arg.matches_rank(ttarg.rank):
                # todo: Use min rank if it differs from rank, in error message
                raise util.error.SemanticError(
                  "{}: argument '{}' has wrong rank, expected rank {}".format(
                    ttfunccall.name,expected_arg.name,
                    ttarg.rank
                  )
                )
        # bytes per element
        if check_bytes_per_element:
            if not expected_arg.matches_any_type:
                expected_bytes_per_element = self._lookup_bytes_per_element1(
                  expected_arg.type,expected_arg.kind 
                )
                if ttarg.bytes_per_element != expected_bytes_per_element:
                    raise util.error.SemanticError(
                      "{}: argument '{}' has wrong kind, expected {} bytes".format(
                        ttfunccall.name,expected_arg.name,
                        expected_bytes_per_element
                      )
                    )
    

    def _resolve_function_call(self,
        ttfunccall,
        symbol_info
      ):
        """Checks the order, type and rank of function call arguments.

        * Check that no positional argument appears after the first keyword argument.
        * Check that required arguments are present.
        * If the function is elemental, arguments type and kind must match with
          the expected argument while ranks must be the same across arguments or scalar.
        * If the function is not elemental, type, kind and rank must match with
          the expected argument.
        """
        found_keyword_arg = False
        dummy_args = symbol_info.dummy_args
        func_result = symbol_info.result
        func_result_type = func_result.type
        func_result_kind = func_result.kind
        present_args = []
        max_rank = 0
        for pos,ttarg in enumerate(ttfunccall.args):
            if isinstance(ttarg,tree.TTIndexRange):
                raise util.error.SyntaxError(
                  "{}: no index range allowed as function call argument".format(
                    ttfunccall.name
                  )
                )
            elif isinstance(ttarg,tree.TTKeywordArgument):
                found_keyword_arg = True
                arg_name = ttarg.key.lower()
                if arg_name not in dummy_args:
                    raise util.error.SemanticError(
                      "{}: keyword '{}' does match any dummy argument name".format(
                         ttfunccall.name,ttarg.key
                       )
                    )
            else:
                if found_keyword_arg:
                    raise util.error.SemanticError(
                      "{}: positional argument after keyword argument".format(
                         ttfunccall.name,ttarg.key
                      )
                    )
                arg_name = dummy_args[pos]
            present_args.append(arg_name)
        
            # compare type, bytes per element, and rank for non-elemental functions
            expected_arg = symbol_info.get_argument(arg_name)
            self._compare_actual_and_expected_argument(
              ttfunccall,ttarg,expected_arg,
              check_rank = not symbol_info.is_elemental, 
              check_bytes_per_element = True
            )
            # func has kind arg that modifies result kind
            if (arg_name.lower() == "kind" 
               and symbol_info.result_depends_on_kind):
                if isinstance(ttarg,tree.TTKeywordArgument):
                    func_result_kind = ttarg.value.fstr().replace(" ","")
                else:
                    func_result_kind = ttarg.fstr().replace(" ","")
            max_rank = max(max_rank,ttarg.rank)
        # set bytes per element
        ttfunccall.bytes_per_element = self._lookup_bytes_per_element1(
          func_result_type,func_result_kind
        )
 
        # check if all non-optional arguments are specified
        self._check_required_arguments_are_present(symbol_info,present_args)
        if symbol_info.is_elemental:
            if max_rank > 0:
                self._check_elemental_function_argument_rank(
                  ttfunccall,symbol_info,max_rank
                )
            if symbol_info.is_conversion:
                self._check_conversion_arguments(ttfunccall,symbol_info)

    def _check_required_arguments_are_present(self,
        symbol_info,
        present_args,
      ):
        """Check if all non-optional arguments are specified.
        :raise util.error.SemanticError: If a required argument is not specified.
        """
        for arg_name in symbol_info.dummy_args:
            expected_arg = symbol_info.get_argument(arg_name)
            if not expected_arg.is_optional:
                if arg_name not in present_args:
                    raise util.error.SemanticError(
                      "{}: required argument '{}' not specified".format(
                         ttfunccall.name,arg_name
                      )
                    )
                    
    def _check_elemental_function_argument_rank(self,
        ttfunccall,
        symbol_info,
        max_rank
      ):
        """Checks that all arguments have the same rank 'max_rank' or are scalars,
        ignoring the kind argument if it is present.
        """
        for name,ttarg in ttfunccall.iter_actual_arguments():
            expected_arg = symbol_info.get_argument(name.lower())
            if not (
                name.lower() == "kind" 
                and ttfunccall.result_depends_on_kind
              ):
                rank = ttarg.rank
                if rank > 0 and rank != max_rank:
                    raise util.error.SyntaxError(
                      "{}: incompatible elemental function argument ranks, {} vs {}".
                        format(
                          ttfunccall.name,
                          max_rank,
                          rank
                        )
                      )

    def _check_conversion_arguments(self,
        ttfunccall,
        symbol_info
      ):
        """Performs the following checks:
        * Checks that the main arguments x and y of
          REAL(x,kind), INT(x,kind), and CMPLX(x,y,kind) are numeric, 
          i.e. REAL,INTEGER,COMPLEX.
        * Checks that the y or x argument of CMPLX(x,y,kind)
          is not a COMPLEX number if both are specified.
        * Checks that the main argument x of
          LOGICAL(x,kind) is LOGICAL itself.
        """
        if symbol_info.name == "cmplx":
            one_arg_is_complex = False
            num_args = 1 
            for name,ttarg in ttfunccall.iter_actual_arguments():
                if not ttarg.yields_numeric_type:
                    raise util.error.SemanticError(
                      "cmplx: '{}' argument is not numeric".format(
                        name
                      ) 
                    )
                if ttarg.yields_complex:
                    one_arg_is_complex = True
                if name != "kind":
                    num_args += 1
            if one_arg_is_complex and num_args > 1:
                raise util.error.SemanticError(
                  "cmplx: two arguments specified while one is complex"
                )
        elif symbol_info.name in ["real","int"]:
            for name,ttarg in ttfunccall.iter_actual_arguments():
                if not ttarg.yields_numeric_type: # kind must yield numeric too
                    raise util.error.SemanticError(
                      "{}: '{}' argument is not numeric".format(
                        symbol_info.name,
                        name
                      ) 
                    )
        elif symbol_info.name == "logical":
            for name,ttarg in ttfunccall.iter_actual_arguments():
                if not ttarg.yields_numeric_type: # kind must yield numeric too
                    raise util.error.SemanticError(
                      "{}: '{}' argument is not logical".format(
                        symbol_info.name,
                        name
                      ) 
                    )
        else:
              raise util.error.LimitationError(
                "handling of conversion '{}' is not implemented".format(ttfunccall.name)
              )
   
    def _resolve_array_expr(self,ttnode):
        ttnode.bytes_per_element = self._lookup_bytes_per_element(ttnode)
        if ttnode.symbol_info.rank != len(ttnode.args):
            raise util.error.SemanticError(
              "'{}': rank mismatch in array reference, expected only {} indices".format(
                ttnode.name,ttnode.symbol_info.rank
              )
            )
        self._check_array_indices(ttnode)
 
    def _resolve_function_call_or_array_expr(self,ttnode,scope,name=None):
        """Resolve a function call or array indexing expression.
        In Fortran, often it is necessary, to rely on the symbol table to distinguish
        a function call from an array indexing expression.
        :param name: A name for the node, so that it can be looked up from within
                     the derived type resolution routine.
        :todo: If the name of the function is an interface name, multiple 
               different function candidates need to be checked. 
               In this case, would return a list of candidates and the 
               argument matching must identify the correct candidate, if any.
               This is currently not implemented.
        """
        if name == None:
            name = ttnode.name
        symbol_info = indexer.scope.search_scope_for_value_expr(scope,name)
        if isinstance(symbol_info,indexer.indexertypes.IndexVariable):
            ttnode.symbol_info = symbol_info
            self._resolve_array_expr(ttnode)
        else:
            ttnode.symbol_info = symbol_info # todo
            # :todo: support checking multiple candidates, think interfaces
            self._resolve_function_call(ttnode,ttnode.symbol_info)
    
    def _resolve_derived_type_expr(self,ttnode,scope):
        """Resolve a derived type lvalue or rvalue expression.
        Assigns each part symbol information from the scope. 
        Ensures that not more than one part of the derived
        type expression has a rank greater than 0.
        :note: The innermost element could be a type-bound procedure
               in contrast to the other derived type expr parts.
        """
        # (a%(b%c))
        identifier_parts = []
        parts_with_nonzero_rank = 0
        for ttpart in ttnode.walk_derived_type_parts_preorder():
            identifier_parts.append(ttpart.derived_type.name)
            var_name = "%".join(identifier_parts)
            if isinstance(ttpart.derived_type,tree.TTFunctionCall):
                ttpart.derived_type.symbol_info = indexer.scope.search_scope_for_var(
                  scope,var_name
                )
                self._resolve_array_expr(ttpart.derived_type) 
            else: #if isinstance(ttpart.derived_type,tree.TTIdentifier):
                self._resolve_identifier(
                  ttpart.derived_type,scope,var_name
                ) 
            if ttpart.derived_type.rank > 0:
                parts_with_nonzero_rank += 1
        # note: ttpart points to innermost part after loop
        identifier_parts.append(ttpart.element.name)
        var_name = "%".join(identifier_parts)
        if isinstance(ttpart.element,tree.TTFunctionCall):
            self._resolve_function_call_or_array_expr(
              ttpart.element,scope,var_name
            ) 
        else: #if isinstance(ttpart.element,tree.TTIdentifier):
            self._resolve_identifier(
              ttpart.element,scope,var_name
            ) 
        if ttpart.element.rank > 0:
            parts_with_nonzero_rank += 1
        if parts_with_nonzero_rank > 1:
            raise util.error.SemanticError("two or more part references with nonzero rank")
    
    def _check_unary_op(self,ttunaryop):
        """Checks that operator and operand type are compatible.
        """
        self._compare_operand_and_operator(ttunaryop,ttunaryop.opd)

    def _resolve_binary_op_chain(self,ttbinaryop):
        """Resolve the type and rank of an binary operation.
        Implemented rules:
        * type in {COMPLEX,REAL,INTEGER}, i.e. numeric type, cannot appear with type in {CHARACTER,LOGICAL}
        * rank1 > 0 can only appear with rank2 in {rank1, 0}
        * COMPLEX(kind1) <op> COMPLEX(kind2) -> COMPLEX(max(kind1,kind2))
        * REAL(kind1) <op> REAL(kind2) -> REAL(max(kind1,kind2))
        * INTEGER(kind1) <op> INTEGER(kind2) -> INTEGER(max(kind1,kind2))
        * LOGICAL(kind1) <op> LOGICAL(kind2) -> LOGICAL(max(kind1,kind2))
        * COMPLEX(kind1) <op> {REAL,INTEGER}(kind2) -> COMPLEX(kind1)
        * REAL(kind1) <op> INTEGER(kind2) -> REAL(kind1)
        """
        opds = ttbinaryop.operands
        result_type = opds[0].type
        result_bytes_per_element = self._lookup_bytes_per_element(opds[0])
        for opd in opds[1:]:
            self._compare_operand_and_operator(ttbinaryop,opd)
            # below is ensured that numeric types are not mixed with logical or character
            (result_type, result_bytes_per_element) = self._compare_type(
              result_type,result_bytes_per_element,
              opd.type,opd.bytes_per_element
            )
        ttbinaryop.rank = self._compare_binary_op_operand_ranks(ttbinaryop.operands)
        ttbinaryop.type = result_type 
        ttbinaryop.bytes_per_element = result_bytes_per_element

    # API
    def resolve_value(self,ttnode,scope):
        """Resolve lvalue and rvalue expressions.
        :param ttnode: The assignment tree node.
        :param scope: Scope for looking up variables and procedures.
        """
        if isinstance(ttnode.value,tree.Literal):
            self._resolve_literal(ttnode.value,scope)
        elif isinstance(ttnode.value,tree.TTIdentifier):
            self._resolve_identifier(ttnode.value,scope)
        elif isinstance(ttnode.value,tree.TTFunctionCall):
            self._resolve_function_call_or_array_expr(ttnode.value,scope)
        elif isinstance(ttnode.value,tree.TTDerivedTypePart):
            self._resolve_derived_type_expr(ttnode.value,scope)
        elif isinstance(ttnode.value,tree.TTArrayConstructor):
            self._check_array_constructor(ttnode.value,scope) 
        elif isinstance(ttnode.value,tree.TTComplexConstructor):
            self._check_complex_constructor(ttnode.value,scope) 

    def resolve_arith_expr(self,ttarithexpr,scope):
        """Resolve an arithmetic expressions and its child nodes.
        :param ttnode: The arithmetic expression  tree node.
        :param scope: Scope for looking up variables and procedures.
        :note: Can be applied to subexpressions of arithmetic expressions too.
        :note: Performs post-order walk to ensure that
              child nodes are resolved before their parents are, e.g.
              rvalues before binary operators.
        """
        for ttnode in ttarithexpr.walk_postorder():
            if isinstance(ttnode,tree.TTValue):
                self.resolve_value(ttnode,scope)
            elif isinstance(ttnode,tree.TTIndexRange):
                self._resolve_index_range(ttnode) 
            elif isinstance(ttnode,tree.TTUnaryOp):
                self._check_unary_op(ttnode)
            elif isinstance(ttnode,tree.TTBinaryOpChain):
                self._resolve_binary_op_chain(ttnode)

    def resolve_assignment(self,ttnode,scope):
        """Resolve the left-hand side (LHS) and right-hand side (RHS) of an assignment.
        :param ttnode: The assignment tree node.
        :param scope: Scope for looking up variables and procedures.
        Further ensure type and rank of LHS and RHS are compatible.
        """
        self.resolve_arith_expr(ttnode.lhs,scope)
        self.resolve_arith_expr(ttnode.rhs,scope)
        lhs_bytes_per_element = ttnode.lhs.bytes_per_element
        rhs_bytes_per_element = ttnode.rhs.bytes_per_element
        if ttnode.lhs.yields_numeric_type:
            if not ttnode.rhs.yields_numeric_type:
                raise util.error.SemanticError(
                  "LHS is numeric but RHS is not numeric"
                )
        elif ttnode.lhs.yields_character:
            if not ttnode.rhs.yields_character:
                raise util.error.SemanticError(
                  "LHS is character but RHS is not character"
                )
        elif ttnode.lhs.yields_logical:
            if not ttnode.rhs.yields_logical:
                raise util.error.SemanticError(
                  "LHS is logical but RHS is not logical"
                )

    def _acc_check_var_is_not_mapped_twice(self,ttaccdir):
        """Checks that no variable appears twice across
        all OpenACC mapping clauses, private and first_private clauses,
        and reduction clauses.
        :note: Must be used after all clauses have been resolved.
        :todo: Currently only symbol based and does not consider
               that different subarrays of the same arrays may be mapped.
        """
        tags = set()
        def add_tag_(tag):
            nonlocal tags
            if tag in tags:
                raise util.error.SemanticError(
                  "variable reference '{}' is subject to multiple mappings".format(tag)
                )
            tags.add(tag) 
        for (ttvarexpr,mapping_kind,readonly) in ttaccdir.walk_mapped_variables():
            add_tag_(ttvarexpr.type_defining_node.symbol_info.tag)
        for ttvarexpr in ttaccdir.walk_private_variables():
            add_tag_(ttvarexpr.type_defining_node.symbol_info.tag)
        for ttvarexpr in ttaccdir.walk_firstprivate_variables():
            add_tag_(ttvarexpr.type_defining_node.symbol_info.tag)
        for (ttvarexpr,reduction_op) in ttaccdir.walk_reduction_variables():
            add_tag_(ttvarexpr.type_defining_node.symbol_info.tag)

    def _acc_resolve_clauses(self,ttaccdir,scope):

        """Resolve the clauses of an OpenACC directive.
        :param ttaccdir: The directive tree node.
        :param scope: Scope for looking up variables and procedures.
        """
        for clause in ttaccdir.walk_clauses():
            if isinstance(clause,(
                tree.TTAccClauseSeq,
                tree.TTAccClauseAuto,
                tree.TTAccClauseIndependent,
                tree.TTAccClauseRead,
                tree.TTAccClauseWrite,
                tree.TTAccClauseCapture,
                tree.TTAccClauseUpdate,
                tree.TTAccClauseNohost,
                tree.TTAccClauseFinalize,
                tree.TTAccClauseIfPresent,
              )):
                pass # no argument, nothing to resolve
            if isinstance(clause,(
                tree.TTAccClauseGang,
                tree.TTAccClauseWorker,
                tree.TTAccClauseVector,
              )):
                if clause.arg_specified:
                    self.resolve_arith_expr(clause.arg,scope)
                    self._check_yields_default_integer_type(
                      clause.arg,
                      "argument to '{}' clause is not default integer kind".format(
                        clause.kind
                      )
                    )
            if isinstance(clause,(
                tree.TTAccClauseNumGangs,
                tree.TTAccClauseNumWorkers,
                tree.TTAccClauseVectorLength,
              )):
                 self.resolve_arith_expr(clause.arg,scope)
                 self._check_yields_default_integer_type(
                   clause.arg,
                   "argument to '{}' clause is not default integer kind".format(
                     clause.kind
                   )
                 )
            if isinstance(clause,(
                tree.TTAccClauseDeviceNum
              )):
                 self.resolve_arith_expr(clause.arg,scope)
                 self._check_yields_default_integer_type(
                   clause.arg,
                   "argument to '{}' clause is not default integer kind".format(
                     clause.kind
                   )
                 )
            if isinstance(clause,(
                tree.TTAccClauseDeviceType
              )):
                for dtype in clause.named_device_types(lower_case=False):
                    if dtype.lower() not in self.acc_supported_device_types:
                        raise util.error.SemanticError(
                          "device type '{}' not supported, must be '{}', or '{}'".format(
                            dtype,"', '".join(self.acc_supported_device_types[0:-1]),
                            self.acc_supported_device_types[-1]
                          )
                        )
                        
            if isinstance(clause,(
                tree.TTAccClauseIf,
                tree.TTAccClauseSelf,
              )):
                 self.resolve_arith_expr(clause.arg,scope)
                 self._check_yields_default_logical_type(
                   clause.arg,
                   "arguments of '{}' clause must be of default logical kind".format(clause.kind)
                 )
            if isinstance(clause,(
                tree.TTAccClauseUpdateSelf,
                tree.TTAccClauseUpdateDevice,
              )):
                for child in clause.child_nodes():
                    self.resolve_value(child,scope)
            if isinstance(clause,(
                tree.TTAccClauseCopy,
                tree.TTAccClauseCopyout,
                tree.TTAccClauseCreate,
                tree.TTAccClauseNoCreate,
                tree.TTAccClausePresent,
                tree.TTAccClauseDeviceptr,
                tree.TTAccClauseAttach,
                tree.TTAccClausePrivate,
                tree.TTAccClauseFirstprivate,
                tree.TTAccClauseUseDevice,
                tree.TTAccClauseCopyin,
              )):
                for child in clause.child_nodes():
                    self.resolve_value(child,scope)
            if isinstance(clause,(
                tree.TTAccClauseDefault
              )):
                pass # nothing to check
            if isinstance(clause,(
                tree.TTAccClauseReduction
              )):
                for child in clause.child_nodes():
                    self.resolve_value(child,scope)
            if isinstance(clause,(
                tree.TTAccClauseBind
              )):
                pass # :todo: allow string in grammar
            if isinstance(clause,(
                tree.TTAccClauseTile
              )):
                for arg in clause.child_nodes():
                    self.resolve_arith_expr(arg,scope)
                    self._check_yields_default_integer_type(
                      arg,"arguments of 'tile' clause must be of default integer kind"
                    )
            if isinstance(clause,(
                tree.TTAccClauseCollapse
              )):
                self.resolve_arith_expr(clause.arg,scope)
                self._check_yields_default_integer_type(
                  arg,"arguments of 'collapse' clause must be of default integer kind"
                )
            if isinstance(clause,(
                tree.TTAccClauseWait
              )):
                self.resolve_arith_expr(clause.devnum,scope)
                self._check_yields_default_integer_type(
                  clause.devnum,"'wait' clause 'devnum' must be of default integer kind"
                )
                for queue in clause.queues:
                    self.resolve_arith_expr(clause.arg,scope)
                    self._check_yields_acc_handle_type(
                      queue,"'wait' clause queues must be integers of 'acc_handle_kind'"
                    )
            if isinstance(clause,(
                tree.TTAccClauseAsync  
              )):
                for queue in clause.queues:
                    self.resolve_arith_expr(clause.arg,scope)
                    self._check_yields_default_integer_type(
                      queue,"'async' clause queues must be integers of 'acc_handle_kind'"
                    )
    
    def resolve_acc_directive(self,ttaccdir,scope):
        """Resolves the clauses appearing on the directive."""
        self._acc_resolve_clauses(ttaccdir,scope)
        if isinstance(ttaccdir,tree.TTAccConstruct):
            self._acc_check_var_is_not_mapped_twice(ttaccdir)
