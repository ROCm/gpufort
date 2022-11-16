# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import util
from gpufort import indexer

from . import tree
from . import opts

def _lookup_bytes_per_element1(ftype,kind):
    kind = kind.lower() if isinstance(kind,str) else None
    try:
        return opts.fortran_type_2_bytes_map[
          ftype.lower()][kind]
    except:
        raise util.error.LookupError(
          "no number of bytes found for type '{}' and kind '{}' in"
          + "'translator.opts.fortran_type_2_bytes_map'".format(ftype,kind)
        )

def _lookup_bytes_per_element(ttnode):
    ttnode.bytes_per_element = _lookup_bytes_per_element1(
      ttnode.type, ttnode.kind
    )

def _resolve_literal(ttnode,scope):
    if isinstance(ttnode,(tree.TTLogical,
                          tree.TTNumber)):
        _lookup_bytes_per_element(ttnode)

def _resolve_identifier(ttnode,scope):
    ttnode.symbol_info = indexer.scope.search_scope_for_var(scope,ttnode.fstr())
    _lookup_bytes_per_element(ttnode)

def _resolve_function_call(ttnode,scope):
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
      scope,ttnode.fstr()
    )
    if ttnode.type != "type":
        _lookup_bytes_per_element(ttnode)
    if ttnode.is_array_expr():
        # rank of the array section expression != array rank
        # Hence we cannot use ttnode.rank
        array_rank = ttnode.symbol_info["rank"]
        if len(ttnode.args) != array_rank:
            raise util.error.SemanticError(
              "{actual} instead of expected {expected} index expressions passed to array variable '{var}'.".format(
                actual=len(ttnode.args),
                expected=array_rank,
                var=ttnode.name 
              )
            )
        for arg in ttnode.args:
            pass        
    else: # is function call
        for arg in ttnode.args:
            pass

def _check_derived_type_part(ttderivedtypemember,scope):
    """ - check that not more than one part reference has nonzero rank, e.g.
          disallows expressions such as "a(:)%b(:)
        - check that element is actually member of the value type
    :note: Assumes that all types have already been resolved."""
    already_found_part_reference_with_nonzero_rank = False
    for ttnode in ttderivedtypemember.walk_derived_type_parts_postorder():
        for child in ttnode.child_nodes():
            if child.rank > 0:
                if already_found_part_reference_with_nonzero_rank:
                    raise error.SemanticError(
                      "Two or more part references with nonzero rank must not be specified"
                    )
                already_found_part_reference_with_nonzero_rank = True

def _check_complex_arith_expr(ttcomplexarithexpr,scope):
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

def _compare_op_and_opd_type(op_type,opd_type):
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

def _check_unary_op(ttnode,scope):
    _compare_op_and_opd_type(
      ttnode.operator_type,
      opd_type
    )

def _resolve_binary_op(ttnode,scope):
    """
    :note: Assumes that all operands have already been resolved."""
    op_type = ttnode.operator_type
    # first operand
    first_opd = ttnode.operands[0]
    current_type = first_opd.type
    current_rank = first_opd.rank
    current_bytes_per_element = first_opd.bytes_per_element
    _compare_op_and_opd_type(op_type,first_opd.type)
    # reamining operands
    for opd in ttnode.operands[1:]:
        _compare_op_and_opd_type(op_type,opd.type)
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

def resolve_arith_expr(ttarithexpr,scope):
    """Resolve the types in an arithmetic expression parse tree.
    Post-order traversal is mandatory as this traverses descedants
    before their parents.
    :raise util.error.LookupError: if a symbol's type could not be determined.
    :note: Can also be applied to subtrees contained in TTArithExpr.
    """
    assert isinstance(ttarithexpr,tree.TTNode), type(ttarithexpr)
    for ttnode in ttarithexpr.walk_postorder():
        if isinstance(ttnode,tree.Literal):
            _resolve_literal(ttnode,scope)
        elif isinstance(ttnode,tree.TTIdentifier):
            _resolve_identifier(ttnode,scope)
        elif isinstance(ttnode,tree.TTFunctionCall):
            _resolve_function_call(ttnode,scope)
        elif isinstance(ttnode,tree.TTValue):
            if isinstance(ttnode._value,tree.TTDerivedTypePart):
                _check_derived_type_part(ttnode._value,scope)
        elif isinstance(ttnode,tree.TTUnaryOp):
            _check_unary_op(ttnode,scope)
        elif isinstance(ttnode,tree.TTBinaryOpChain):
            _resolve_binary_op(ttnode,scope)
