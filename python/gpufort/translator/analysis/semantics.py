# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import util
from gpufort import indexer

from . import tree

def _check_derived_type_member(ttderivedtypemember,scope):
    """Check that not more than one part reference has nonzero rank, e.g.
    disallows expressions such as "a(:)%b(:)
    :note: Assumes that all types have already been resolved."""
    already_found_part_reference_with_nonzero_rank = False 
    for ttnode in ttderivedtypemember.walk_derived_type_members_postorder():
        for child in ttnode.child_nodes():
            if child.rank() > 0:
                if already_found_part_reference_with_nonzero_rank:
                    raise error.SemanticError(
                      "Two or more part references with nonzero rank must not be specified"
                    )
                already_found_part_reference_with_nonzero_rank = True

def _check_binary_op(ttbinaryop,scope):
    """
    
    :note: Assumes that all operands have already been resolved."""
    already_found_part_reference_with_nonzero_rank = False 
    for ttnode in ttderivedtypemember.operands():

def resolve_arith_expr(ttarithexpr,scope):
    """Resolve the types in an arithmetic expression parse tree.
    :raise util.error.LookupError: if a symbol's type could not be determined.
    :note: Can also be applied to subtrees contained in TTArithExpr.
    """
    for ttnode in ttarithexpr.walk_postorder():
        if isinstance(ttnode,tree.TTIdentifier):
            ttnode.irecord = indexer.scope.search_scope_for_var(scope,ttnode.fstr())
        elif isinstance(ttnode,tree.TTTensorEval):
            ttnode.irecord = indexer.scope.search_scope_for_value_expr(scope,ttnode.fstr())
        elif isinstance(ttnode,tree.TTValue):
            if isinstance(ttnode._value,tree.TTDerivedTypeMember):
                _check_derived_type_member(ttnode._value,scope)
            
