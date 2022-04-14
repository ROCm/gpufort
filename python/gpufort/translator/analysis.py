# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
"""This module contains functions for 
determinig the input and output variables
of a loop nest or procedure that is mapped
to the device.
"""
import copy
import pyparsing

from gpufort import indexer
from gpufort import util

from . import tree
from . import conv
from . import opts

def append_c_type(tavar):
    f_type = tavar["f_type"]
    kind = tavar["kind"]
    if f_type == "type":
        tavar["c_type"] = tavar["kind"] 
    elif f_type == "character":
        tavar["c_type"] = "char"
        # TODO more carefully check if len or kind is specified for characters
    elif f_type != "character": 
        tavar["c_type"] = conv.convert_to_c_type(f_type, kind, "TODO unknown")
    tavar["bytes_per_element"] = conv.num_bytes(f_type, kind, default=None)

def _create_analysis_var(scope, var_expr):
    ivar = indexer.scope.search_scope_for_var(
        scope, var_expr)
    tavar = copy.deepcopy(ivar)
    tavar["expr"] = util.parsing.strip_array_indexing(var_expr).lower()
    tavar["c_name"] = tavar["name"] 
    tavar["op"]   = ""
    append_c_type(tavar)
    return tavar

def _lookup_index_vars(scope, var_exprs, consumed_var_exprs=[]):
    """Search scope for index vars and remove corresponding 
       var expression from all_vars2 list."""
    tavars = []
    for var_expr in var_exprs:
        tavar = _create_analysis_var(scope,var_expr)
        tavars.append(tavar)
        consumed_var_exprs.append(var_expr)
    return tavars

def lookup_index_entries_for_vars_in_kernel_body(scope,
                                                 all_vars,
                                                 reductions,
                                                 shared_vars,
                                                 local_vars,
                                                 loop_vars,
                                                 explicitly_mapped_vars=[]):
    """Lookup index variables
    :param list all_vars: List of all variable expressions (var)
    :param list reductions: List of tuples pairing a reduction operation with the associated
                             variable expressions
    :param list shared:     List of variable expressions that are shared by the workitems/threads in a workgroup/threadblock
    :param list local_vars: List of variable expressions that can be mapped to local variables per workitem/thread
    :param list explicitly_mapped_vars: List of explicitly mapped vars.
    :note: Emits errors (or warning) if a variable in another list is not present in all_vars
    :note: Emits errors (or warning) if a reduction variable is part of a struct.
    :return: A tuple containing (in this order): global variables, reduced global variables, shared variables, local variables
             as list of index entries. The reduced global variables have an extra field 'op' that
             contains the reduction operation.
    """
    consumed = []
    talocal_vars = _lookup_index_vars(
        scope,
        local_vars, consumed)
    tashared_vars = _lookup_index_vars(
        scope,
        shared_vars, consumed)
    all_vars2 = [
        v for v in all_vars
        if not v in consumed and v not in loop_vars
    ]

    taglobal_reduced_vars = []
    taglobal_vars = []

    for reduction_op, var_exprs in reductions.items():
        for var_expr in var_exprs:
            if "%" in var_expr:
                raise util.error.LimitationError("reduction of derived type members not supported")
            else:
                tavar1 = _create_analysis_var(scope,var_expr)
                if tavar1["rank"] > 0:
                    raise util.error.LimitationError("reduction of arrays or array members not supported")
                tavar = copy.deepcopy(tavar1)
                tavar["op"] = conv.get_operator_name(reduction_op)
                taglobal_reduced_vars.append(tavar)
            try:
                all_vars2.remove(var_expr)
            except:
                pass # TODO error

    for var_expr in all_vars2:
        tavar = _create_analysis_var(scope, var_expr)
        taglobal_vars.append(tavar)
    return taglobal_vars, taglobal_reduced_vars, tashared_vars, talocal_vars

def _apply_substitions(tavars,substitutions):
    for i,tavar in enumerate(tavars):
        var_expr = tavar["expr"]
        if var_expr in substitutions:
            tavar["c_name"] = substitutions[var_expr]

def lookup_index_entries_for_vars_in_loopnest(scope,ttloopnest,loop_vars,substitutions={}):
    all_vars       = vars_in_subtree(ttloopnest, scope)
    local_scalars_ = local_scalars(ttloopnest,scope)
    local_vars     = [v for v in (local_scalars_+ttloopnest.gang_team_private_vars())
                     if v not in loop_vars]
    local_vars    += [v for v in all_vars 
                     if (v[0]=="_" 
                        and v not in loop_vars 
                        and v not in local_vars)]
    # TODO improve
    result = lookup_index_entries_for_vars_in_kernel_body(scope,
                                                          all_vars,
                                                          ttloopnest.gang_team_reductions(),
                                                          ttloopnest.gang_team_shared_vars(),
                                                          local_vars,
                                                          loop_vars)
    for i in range(0,4):
        _apply_substitions(result[i],substitutions)
    return result

def lookup_index_entries_for_vars_in_procedure_body(scope,ttprocedurebody,iprocedure):
    shared_vars = [
        ivar["name"]
        for ivar in iprocedure["variables"]
        if "shared" in ivar["qualifiers"]
    ]
    local_vars = [
        ivar["name"]
        for ivar in iprocedure["variables"]
        if ivar["name"] not in iprocedure["dummy_args"]
    ]
    all_var_exprs = vars_in_subtree(ttprocedurebody, scope) # in the body, there might be variables present from used modules
    all_vars = iprocedure["dummy_args"] + [
        v for v in all_var_exprs if (v not in iprocedure["dummy_args"] and
                                     v not in tree.grammar.DEVICE_PREDEFINED_VARIABLES)
    ]
    global_reductions = {}
    loop_vars = []

    return lookup_index_entries_for_vars_in_kernel_body(scope,
                                                        all_vars,
                                                        global_reductions,
                                                        shared_vars,
                                                        local_vars,
                                                        loop_vars)

def get_kernel_args(taglobal_vars, taglobal_reduced_vars, tashared_vars, talocal_vars):
    """:return: index records for the variables
                that must be passed as kernel arguments.
    :note: Shared_vars and local_vars must be passed as kernel argument
           as well if the respective variable is an array. 
    """
    return (taglobal_vars
            + taglobal_reduced_vars
            + [tavar for tavar in tashared_vars if tavar["rank"] > 0]
            + [tavar for tavar in talocal_vars if tavar["rank"] > 0])
    
def find_all_matching_exclude_directives(ttnode,
                                         filter_expr=lambda x: True):
    """Find all translator tree nodes that match the filter_expr.
    Does not search subtrees of directives."""
    result = []

    def traverse_(curr):
        if isinstance(curr, tree.ILoopAnnotation):
            return
        if filter_expr(curr):
            result.append(curr)
        if isinstance(curr,pyparsing.ParseResults) or\
           isinstance(curr,list):
            for el in curr:
                traverse_(el)
        elif isinstance(curr, tree.TTNode):
            for el in curr.children():
                traverse_(el)

    traverse_(ttnode)
    return result

def search_lrvalue_exprs_in_subtree(ttnode, search_filter, scope, min_rank=-1):
    """All lvalue and ralue exprs in the subtree.
    :note: Returns simplified expressions, stripped off all array indexing syntax.
    :note: Excludes device predefined variables such as `threadidx%x` etc.
    """
    tags = []
    for ttvalue in find_all_matching_exclude_directives(
            ttnode.body,
            search_filter): # includes the identifiers of the function calls 
        # TODO Search tag creation strips array brackets
        # TODO provide clean interface to value
        tag = indexer.scope.create_index_search_tag_for_var(ttvalue._value.f_str()) # TODO 
        if tag not in tree.grammar.DEVICE_PREDEFINED_VARIABLES:
            ivar = indexer.scope.search_scope_for_var(scope, tag)
            if ivar["rank"] >= min_rank and\
               not tag in tags: # ordering important
                tags.append(tag)
    return tags


def vars_in_subtree(ttnode, scope):
    """:return: all identifiers of LValue and RValues in the body."""

    def search_filter(node):
        cond1 = (isinstance(node,tree.IValue)
                and isinstance(node._value, (tree.TTDerivedTypeMember,tree.TTIdentifier,tree.TTFunctionCallOrTensorAccess)))
        if cond1 and isinstance(node._value, tree.TTFunctionCallOrTensorAccess):
            return node._value.is_tensor()
        else:
            return cond1 
    result = search_lrvalue_exprs_in_subtree(ttnode, search_filter, scope)
    return result

def arrays_in_subtree(ttnode, scope):

    def search_filter(node):
        cond1 = (isinstance(node,tree.IValue) 
                and isinstance(node._value, tree.TTFunctionCallOrTensorAccess))
        if cond1 and isinstance(node._value, tree.TTFunctionCallOrTensorAccess):
            return node._value.is_tensor()
        else:
            return cond1 

    return search_lrvalue_exprs_in_subtree(ttnode, search_filter, scope, 1)


def inout_arrays_in_subtree(ttnode, scope):

    def search_filter(node):
        cond1 = (isinstance(node,tree.TTLValue) 
                and isinstance(node._value, tree.TTFunctionCallOrTensorAccess))
        if cond1 and isinstance(node._value, tree.TTFunctionCallOrTensorAccess):
            return node._value.is_tensor()
        else:
            return cond1 

    return search_lrvalue_exprs_in_subtree(ttnode, search_filter, scope, 1)

#def all_unmapped_arrays(ttloopnest, scope):
#    """:return: Name of all unmapped array variables"""
#    mapped_vars = ttloopnest.all_mapped_vars()
#    if ttloopnest.all_arrays_are_on_device():
#        return []
#    else:
#        return [var for var in arrays_in_body if not var in mapped_vars]
#
#def deviceptrs(ttloopnest):
#    if ttloopnest.all_arrays_are_on_device():
#        return arrays_in_subtree(ttloopnest, scope)
#    else:
#        return ttloopnest.deviceptrs()
    
def local_scalars_and_reduction_candidates(ttloopnest, scope):
    """
    local variable      - scalar variable that is not read before the assignment (and is no derived type member)
    reduction_candidates - scalar variable that is written but not read anymore 
    NOTE: Always returns Fortran identifiers
    NOTE: The loop variables need to be removed from this result when rendering the corresponding C kernel.
    NOTE: Implementatin assumes that loop condition variables are not written to in loop body. 
    NOTE: When rendering the kernel, it is best to exclude all variables for which an array declaration has been found,
    from the result list. TTCufKernelDo instances do not know of the type of the variables.
    """
    scalars_read_so_far = [
    ] # per line, with name of lhs scalar removed from list
    initialized_scalars = []
    # depth first search
    
    assignments = find_all_matching_exclude_directives(
        ttloopnest.body[0], lambda node: isinstance(node,(
            tree.TTAssignment, tree.TTComplexAssignment, tree.
            TTMatrixAssignment
        )))
    for assignment in assignments:
        # lhs scalars
        lvalue = assignment._lhs
        lvalue_name = lvalue._value.f_str().lower()
        if isinstance(lvalue, tree.TTIdentifier): # could still be a matrix
            ivar = indexer.scope.search_scope_for_var(
                scope, lvalue_name)
            if ivar["rank"] == 0 and\
               not lvalue_name in scalars_read_so_far:
                initialized_scalars.append(
                    lvalue_name) # read and initialized in
        # rhs scalars
        ttrvalues = tree.find_all(assignment._rhs,tree.TTRValue)
        for ttrvalue in ttrvalues:
            if isinstance(ttrvalue._value,tree.TTIdentifier):
                rvalue_name = ttrvalue._value.f_str().lower()
                ivar = indexer.scope.search_scope_for_var(
                    scope, rvalue_name)
                if ivar["rank"] == 0 and\
                   rvalue_name != lvalue_name: # do not include name of rhs if lhs appears in rhs
                    scalars_read_so_far.append(rvalue_name)
    # initialized scalars that are not read (except in same statement) are likely reductions
    # initialized scalars that are read again in other statements are likely local variables
    reduction_candidates = [
        name for name in initialized_scalars
        if name not in scalars_read_so_far
    ]
    local_scalars = [
        name for name in initialized_scalars
        if name not in reduction_candidates
    ] # contains loop variables
    return local_scalars, reduction_candidates

def local_scalars(ttloopnest, scope):
    local_scalars, _ = local_scalars_and_reduction_candidates(ttloopnest, scope)
    return local_scalars

def reduction_candidates(ttloopnest, scope):
    _, reduction_candidates = local_scalars_and_reduction_candidates(ttloopnest, scope)
    return reduction_candidates

def loop_vars_in_loopnest(ttdos):
    identifier_names = []
    for ttdo in ttdos:
        identifier_names.append(ttdo.loop_var(tree.make_f_str))
    return identifier_names

def _perfectly_nested_outer_do_loops(ttloopnest):
    """TODO 
    """
    if isinstance(ttloopnest.body[0],tree.TTStatement):
        # in this case, an array expression was unpacked.
        # The result is always a perfect loopnest
        return tree.find_all(ttloopnest.body[0], tree.TTDo), True
    elif isinstance(ttloopnest.body[0],tree.TTDo):
        ttdos = []
        # check for interdependent loops
        current = ttloopnest.body[0]
        ttdos.append(current)
        
        def condition_(current):
            return (len(current.body) == 1
                   and (isinstance(current.body[0],tree.TTDo)
                       or (isinstance(current.body[0],tree.TTStatement)
                          and isinstance(current.body[0]._statement,tree.TTDo))))
        while condition_(current):
            if isinstance(current.body[0],tree.TTDo):
                current = current.body[0]
            else:
                current = current.body[0]._statement
            ttdos.append(current)
        return ttdos, False

def perfectly_nested_do_loops_to_map(ttloopnest):
    num_loops_to_map = max(1,int(ttloopnest.parent_directive().num_collapse()))
    ttdos, from_array_expression = _perfectly_nested_outer_do_loops(ttloopnest)
    if from_array_expression:
        num_loops_to_map = len(ttdos)
    return ttdos[0:num_loops_to_map]

def problem_size(ttdos,**kwargs):
    loop_collapse_strategy,_ = util.kwargs.get_value("loop_collapse_strategy",opts.loop_collapse_strategy,**kwargs)
    
    num_loops_to_map = len(ttdos)
    if loop_collapse_strategy == "grid" or num_loops_to_map == 1:
        num_loops_to_map = min(3, num_loops_to_map)
        result = ["-1"] * num_loops_to_map
        for i, ttdo in enumerate(ttdos):
            if i < num_loops_to_map:
                result[i] = ttdo.problem_size()
        return result
    else: # "collapse"
        result = []
        for ttdo in ttdos:
            result.append(ttdo.problem_size())
        if len(result):
            return ["*".join(result)]
        else:
            return ["-1"]

# OpenACC specific

def kernel_args_to_acc_mappings_no_types(acc_clauses,tavars,present_by_default,callback,**kwargs):
    r"""Derive mappings for arrays

    :return: A list of tuples that maps the 'expr' field of a translator analysis var ('tavar')
             to an implementation-specific OpenACC mapping. The implementation-specific
             mapping is realised via a callback.
    
    :param list acc_clauses:        The acc clauses as obtained via util.parsing.parse_acc_clauses(...)
    :param list tavars:             list of translator analysis vars
    :param bool present_by_default: If array variables without explicit mappings are present
                                    by default.
    :param callable callback:       A callback that takes the OpenACC clause as first
                                    argument, the expression subject to the mapping as second, 
                                    and the (implementation-specific) keyword arguments as third argument. 
  
    :param \*\*kwargs: Keyword args to pass to the callback.
    
    :note: Current implementation does not support mapping of scalars/arrays of derived type.
    
    :raise util.error.LimitationError: If a derived type must be mapped.
    :raise util.parsing.SyntaxError: If the OpenACC clauses could not be parsed. 
    """
    mappings = []
    for tavar in tavars:
        if tavar["rank"] > 0: # map array
            explicitly_mapped = False
            for clause in acc_clauses:
                kind1, args = clause
                kind = kind1.lower()
                for var_expr in args:
                    var_tag = util.parsing.strip_array_indexing(var_expr.lower())
                    if tavar["expr"] == var_tag:
                        if tavar["f_type"]=="type":
                            raise util.error.LimitationError("mapping of derived type scalars and arrays not supported (yet)")
                        else:
                            mappings.append((var_expr, callback(kind,var_expr,tavar,**kwargs)))
                            explicitly_mapped = True
                            break
                if explicitly_mapped: break
            if not explicitly_mapped and present_by_default:
                if tavar["f_type"]=="type": # TODO refine
                    raise util.error.LimitationError("mapping of derived type scalars and arrays not supported (yet)")
                else:
                    mappings.append((tavar["expr"], callback("present_or_copy",tavar["expr"],tavar,**kwargs)))
            elif not explicitly_mapped:
                raise util.parsing.SyntaxError("no mapping specified for expression: {}".format(tavar["expr"]))
        elif tavar["f_type"] == "type": # map type
            raise util.error.LimitationError("mapping of derived types and their members not supported (yet)")
        else: # basic type scalars
            mappings.append((tavar["expr"],tavar["expr"]))
    return mappings 