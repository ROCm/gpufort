# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import util
from gpufort import indexer

from . import base
from . import fortran

#def convert_to_do_loop_nest_if_necessary(ttassignment,scope,fortran_style_tensors=True):
#
#    """
#    - step through bounds
#    - if type is range
#      - get upper bound
#        - not found: use a_lb<i>+a_n<i>-1
#      - get lower bound
#         - not found: use a_lb<i>
#    """
#    name = self._lhs.name()
#    loop_nest_start = ""
#    loop_nest_end = ""
#    original_f_str = self.f_str()
#    # TODO fix should be moved into parser preprocessing stage
#    for i, ttrange in enumerate(self._lhs.bounds()):
#        loop_var_name = "_" + chr(
#            ord('a') + i) # Fortran names cannot start with "_"
#        ttrange.set_loop_var(loop_var_name)
#        lbound = ttrange.l_bound()
#        if not len(lbound):
#            if fortran_style_tensors:
#                lbound = "lbound({name},{i})".format(name=name, i=i + 1)
#            else:
#                lbound = "{name}_lb{i}".format(name=name, i=i + 1)
#        ubound = ttrange.u_bound()
#        if not len(ubound):
#            if fortran_style_tensors:
#                ubound = "ubound({name},{i})".format(name=name, i=i + 1)
#            else:
#                ubound = "({ubound} + {name}_n{i} - 1)".format(ubound=ubound,
#                                                               name=name,
#                                                               i=i + 1)
#        stride = ttrange.stride()
#        if not len(stride):
#            stride = "1"
#        loop_nest_start = "do {var}={lb},{ub},{step}\n".format(
#            var=loop_var_name, lb=lbound, ub=ubound,
#            step=stride) + loop_nest_start
#        loop_nest_end = "end do\n" + loop_nest_end
#    if len(loop_nest_start):
#        # TODO put into options
#        for rvalue in base.find_all(self._rhs, searched_type=TTRValue):
#            for i, ttrange in enumerate(rvalue.bounds()):
#                loop_var_name = "_" + chr(
#                    ord('a') + i) # Fortran names cannot start with "_"
#                ttrange.set_loop_var(loop_var_name)
#        loop_nest_start += "! TODO(gpufort) please check if this conversion was done correctly\n"
#        loop_nest_start += "! original: {}\n".format(original_f_str)
#        result = loop_nest_start + self.f_str() + "\n" + loop_nest_end
#        return do_loop.parseString(result)[0]
#    else:
#        return self
    
def _collect_ranges(function_call_args,include_none_values=False) 
        ttranges = []
        for i,enumerate(ttnode in function_call_args):
            if isinstance(ttnode, fortran.TTRange):
                ttranges.append(ttnode)
            elif include_none_values:
                ttranges.append(None)
        return ttranges

def _collect_ranges_in_lrvalue(ttnode,include_none_values=False):
    """
    :return A list of range objects. If the list is empty, no function call
            or tensor access has been found. If the list contains a None element
            this implies that a function call or tensor access has been found 
            was scalar index argument was used.
    """
    current = ttnode
    if isinstance(current,fortran.TTFunctionCallOrTensorAccess):
        return collect_ranges_(current,include_none_values)
    elif isinstance(current,fortran.TTDerivedTypeMember):
        result = []
        while current == TTDerivedTypeMember:
            if isinstance(current._type,fortran.TTFunctionCallOrTensorAccess):
                result += collect_ranges_(current._type._args,include_none_values)
            if isinstance(current._element,fortran.TTFunctionCallOrTensorAccess):
                result += collect_ranges_(current._element._args,include_none_values)
            current = current._element
        return result

def _create_do_loop_statements_fstr(ranges,loop_indices):
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
        indent = "  "*(i-1) 
        do_statements.append("{indent}do {var}={lb},{ub}{step}\n".format(
            indent=indent,
            var=loop_idx, lb=lbound, ub=ubound,
            step=stride))
        end_do_statements.insert(0,"{indent}end do\n")
    return do_statements, end_do_statements

def expand_array_expression(ttassignment,scope,int_counter,fortran_style_tensors=True):
    """
    - step through bounds
    - if type is range
      - get upper bound
        - not found: use a_lb<i>+a_n<i>-1
      - get lower bound
         - not found: use a_lb<i>
    """
    lvalue_f_str = base.make_f_str(ttassignment._lhs.value)
    livar = indexer.scope.search_scope_for_var(scope,lvalue_f_str)
    loop_indices = 
    if livar["rank"] > 0:
        lvalue_ranges = _collect_ranges_in_lrvalue(ttassignment._lhs.value)
        if len(lvalue_ranges):
            loop_indices = [ "".join(["_i",str(int_counter+i)]) for i in range(0,len(lvalue_ranges)) ]
            for i,idx in enumerate(loop_indices):
                lvalue_ranges[i].overwrite_f_str(idx)
            rvalue_ranges = []
            for ttrvalue in base.find_all(ttassignment._rhs, searched_type=TTRValue):
                rvalue_f_str = base.make_f_str(ttassignment._rhs.value)
                rvalue_ranges = _collect_ranges_in_lrvalue(ttrvalue)
                rivar = indexer.scope.search_scope_for_var(scope,rvalue_f_str)            
                if len(lvalue_ranges) == len(rvalue_ranges):
                    for i,idx in enumerate(loop_indices):
                        lvalue_ranges[i].overwrite_f_str(idx)
                else:
                    raise util.error.LimitationError("failed to expand colon operator expression to loopnest: not enough colon expressions in rvalue argument list")
        do_loop_statements, end_do_statements = _create_do_loop_statements_fstr)
        int_counter += len(loop_indices)
        return modified_expression, do_loop_statements, end_do_statements, int_counter += len(loop_indices), True
    else:
        return None, [], [], int_counter, False

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
