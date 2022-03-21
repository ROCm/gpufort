# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import util
from gpufort import indexer

from . import base
from . import fortran

def move_statements_into_loopnest_body(ttloopnest):
    # subsequent loop ranges must not depend on LHS of assignment
    # or inout, out arguments of function call
    pass 

def collapse_loopnest(ttdos):
    indices = []
    problem_sizes = []
    for ttdo in ttdos:
        problem_sizes.append("({})".format(
            ttdo.problem_size(converter=base.make_c_str)))
        ttdo.thread_index = "_remainder,_denominator" # side effects
        indices.append(ttdo.collapsed_loop_index_c_str())
    conditions = [ ttdos[0].hip_thread_bound_c_str() ]
    indices.insert(0,"int _denominator = {};\n".format("*".join(problem_sizes)))
    indices.insert(0,"int _remainder = __gidx1;\n")
    return indices, conditions

def map_loopnest_to_grid(ttdos,num_outer_loops_to_map):
    if num_outer_loops_to_map > 3:
        util.logging.log_warn(
            "ttdo collapse strategy 'grid' chosen with nested loops > 3")
    num_outer_loops_to_map = min(3, num_outer_loops_to_map)
    thread_indices = ["x", "y", "z"]
    for i in range(0, 3 - num_outer_loops_to_map):
        thread_indices.pop()
    indices = []
    conditions = []
    for ttdo in ttdos:
        if not len(thread_indices):
            break
        ttdo.thread_index = thread_indices.pop()
        indices.append(ttdo.hip_thread_index_c_str())
        conditions.append(ttdo.hip_thread_bound_c_str())
    return indices, conditions

def map_allocatable_pointer_derived_type_members_to_flat_arrays(lrvalues,loop_vars,scope):
    r"""Converts derived type expressions whose innermost element is an array of allocatable or pointer
    type to flat arrays in expectation that these will be provided
    to kernel as flat arrays too.
    :return: A dictionary containing substitutions for the identifier part of the 
             original variable expressions.
    """
    substitutions = {}
    for lrvalue in lrvalues:
        ttnode = lrvalue.get_value()
        if isinstance(ttnode,fortran.TTDerivedTypeMember):
            ident = ttnode.identifier_f_str()
            ivar = indexer.scope.search_scope_for_var(scope,ident)
            if (ivar["rank"] > 0
                and ("allocatable" in ivar["qualifiers"]
                    or "pointer" in ivar["qualifiers"])):
                # TODO 
                # search through the subtree and ensure that only
                # the last element is an array indexed by the loop
                # variables 
                # Deep copy required for expressions that do not match
                # this criterion
                if (":" in ident 
                   or "(" in ident): # TODO hack
                    return util.error.LimitationError("cannot map expression '{}'".format(ident))
                var_expr = util.parsing.strip_array_indexing(ident)
                c_name = util.parsing.mangle_fortran_var_expr(var_expr) 
                substitute = ttnode.f_str().replace(ident,c_name)
                ttnode.overwrite_c_str(substitute)
                substitutions[var_expr] = c_name
    return substitutions

def flag_tensors(lrvalues, scope):
    """Clarify types of function calls / tensor access that are not members of a struct."""
    for value in lrvalues:
        if isinstance(value._value, fortran.TTFunctionCallOrTensorAccess):
           try:
              _ = indexer.scope.search_scope_for_var(scope, value.f_str()) # just check if the var exists
              value._value._is_tensor_access = base.True3
           except util.error.LookupError:
              pass
