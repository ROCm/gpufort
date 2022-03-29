# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import util
from gpufort import indexer

from . import tree
from . import analysis
from . import parser

def _collect_ranges(function_call_args,include_none_values=False):
        ttranges = []
        for i,elem in enumerate(function_call_args):
            ttnode = elem[0]
            if isinstance(ttnode, tree.TTRange):
                ttranges.append(ttnode)
            elif include_none_values:
                ttranges.append(None)
        return ttranges

def _collect_ranges_in_lrvalue(lrvalue,include_none_values=False):
    """
    :return A list of range objects. If the list is empty, no function call
            or tensor access has been found. If the list contains a None element
            this implies that a function call or tensor access has been found 
            was scalar index argument was used.
    """
    current = lrvalue._value
    if isinstance(current,tree.TTFunctionCallOrTensorAccess):
        return _collect_ranges(current._args,include_none_values)
    elif isinstance(current,tree.TTDerivedTypeMember):
        result = []
        while current == tree.TTDerivedTypeMember:
            if isinstance(current._type,tree.TTFunctionCallOrTensorAccess):
                result += _collect_ranges(current._type._args,include_none_values)
            if isinstance(current._element,tree.TTFunctionCallOrTensorAccess):
                result += _collect_ranges(current._element._args,include_none_values)
            current = current._element
        return result

def _create_do_loop_statements(ranges,loop_indices,fortran_style_tensors):
    do_statements     = []
    end_do_statements = []
    for i, ttrange in enumerate(ranges,1):
        loop_idx = loop_indices[i-1]
        lbound = ttrange.l_bound()
        if not len(lbound):
            if fortran_style_tensors:
                lbound = "lbound({name},{i})".format(name=name, i=i)
            else:
                lbound = "{name}_lb{i}".format(name=name, i=i)
        ubound = ttrange.u_bound()
        if not len(ubound):
            if fortran_style_tensors:
                ubound = "ubound({name},{i})".format(name=name, i=i)
            else:
                ubound = "({ubound} + {name}_n{i} - 1)".format(ubound=ubound,
                                                               name=name,
                                                               i=i)
        stride = ttrange.stride()
        if len(stride):
            stride = "," + stride
        do_statements.insert(0,"do {var}={lb},{ub}{step}".format(
            var=loop_idx, lb=lbound, ub=ubound,
            step=stride))
        end_do_statements.insert(0,"end do")
    return do_statements, end_do_statements

def _expand_array_expression(ttassignment,scope,int_counter,fortran_style_tensors):
    """
    - step through bounds
    - if type is range
      - get upper bound
        - not found: use a_lb<i>+a_n<i>-1
      - get lower bound
         - not found: use a_lb<i>
    """
    lvalue_f_str = tree.make_f_str(ttassignment._lhs._value)
    livar = indexer.scope.search_scope_for_var(scope,lvalue_f_str)
    loop_indices = [] 
    if livar["rank"] > 0:
        lvalue_ranges = _collect_ranges_in_lrvalue(ttassignment._lhs)
        if len(lvalue_ranges):
            loop_indices = [ "".join(["_i",str(int_counter+i)]) for i in range(0,len(lvalue_ranges)) ]
            for i,idx in enumerate(loop_indices):
                lvalue_ranges[i].overwrite_f_str(idx)
            rvalue_ranges = []
            for ttrvalue in tree.find_all(ttassignment._rhs, searched_type=tree.TTRValue):
                try:
                    rvalue_f_str = tree.make_f_str(ttrvalue._value)
                    rivar = indexer.scope.search_scope_for_var(scope,rvalue_f_str)
                    if rivar["rank"] > 0:
                        rvalue_ranges = _collect_ranges_in_lrvalue(ttrvalue)
                        if len(rvalue_ranges):
                            if len(lvalue_ranges) == len(rvalue_ranges):
                                for i,idx in enumerate(loop_indices):
                                    rvalue_ranges[i].overwrite_f_str(idx)
                            else:
                                raise util.error.LimitationError("failed to expand colon operator expression to loopnest: not enough colon expressions in rvalue argument list")
                except util.error.LookupError:
                    pass          
            do_loop_statements, end_do_statements = _create_do_loop_statements(lvalue_ranges,loop_indices,fortran_style_tensors)
            statements = do_loop_statements + [ttassignment.f_str()] + end_do_statements
            return statements, int_counter + len(loop_indices), True
        else:
            return [], int_counter, False
    else:
        return [], int_counter, False

def expand_all_array_expressions(ttnode,scope,fortran_style_tensors=True):
    def isassignmentstatement_(ttnode):
        return (isinstance(ttnode,tree.TTStatement) 
               and isinstance(ttnode._statement,tree.TTAssignment))
    statements = analysis.find_all_matching_exclude_directives(ttnode,isassignmentstatement_)
    int_counter = 1
    for ttstatement in statements:
        ttassignment = ttstatement._statement
        statements, int_counter, modified =\
          _expand_array_expression(ttassignment,scope,int_counter,
                                   fortran_style_tensors)
        if modified:
            ttdo = parser.parse_fortran_code(statements).body[0] # parse returns ttroot
            ttstatement._statement = ttdo
    return int_counter

def move_statements_into_loopnest_body(ttloopnest):
    # TODO
    # subsequent loop ranges must not depend on LHS of assignment
    # or inout, out arguments of function call
    pass 

def collapse_loopnest(ttdos):
    indices = []
    problem_sizes = []
    for ttdo in ttdos:
        problem_sizes.append("({})".format(
            ttdo.problem_size(converter=tree.make_c_str)))
        ttdo.thread_index = "_remainder,_denominator" # side effects
        indices.append(ttdo.collapsed_loop_index_c_str())
    conditions = [ ttdos[0].hip_thread_bound_c_str() ]
    indices.insert(0,"int _denominator = {};\n".format("*".join(problem_sizes)))
    indices.insert(0,"int _remainder = __gidx1;\n")
    return indices, conditions

def map_loopnest_to_grid(ttdos):
    thread_indices = ["x", "y", "z"]
    while len(thread_indices) > len(ttdos):
        thread_indices.pop()
    indices = []
    conditions = []
    for ttdo in ttdos:
        ttdo.thread_index = thread_indices.pop()
        indices.append(ttdo.hip_thread_index_c_str())
        conditions.append(ttdo.hip_thread_bound_c_str())
        if not len(thread_indices):
            break
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
        if isinstance(ttnode,tree.TTDerivedTypeMember):
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
        if isinstance(value._value, tree.TTFunctionCallOrTensorAccess):
           try:
              _ = indexer.scope.search_scope_for_var(scope, value.f_str()) # just check if the var exists
              value._value._is_tensor_access = tree.True3
           except util.error.LookupError:
              pass
