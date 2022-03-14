# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
"""This module contains functions for 
determinig the input and output variables
of a loop nest or procedure that is mapped
to the device.
"""
import copy

from gpufort import indexer
from . import tree

# TODO only strip certain derived type expressions, e.g. if a struct is copied via copy(<struct>)
# or if a CUDA Fortran derived type has the device attribute

def strip_member_access(var_exprs):
    """Strip off member access parts from derived type member access expressions,
       e.g. 'a%b%c' becomes 'a'.
    """
    # TODO only strip certain derived type expressions, e.g. if a struct is copied via copy(<struct>)
    # or if a CUDA Fortran derived type has the device attribute
    result = []
    return [var_expr.split("%", maxsplit=1)[0] for var_expr in var_exprs]

def _create_analysis_var(scope, var_expr):
    ivar = indexer.scope.search_scope_for_var(
        scope, var_expr)
    tavar = copy.deepcopy(ivar)
    tavar["expr"] = util.parsing.strip_array_indexing(var_expr).lower()
    tavar["op"]   = ""
    return tavar

def _lookup_index_vars(scope, var_exprs, consumed_var_exprs=[]):
    """Search scope for index vars and remove corresponding 
       var expression from all_vars2 list."""
    tavars = []
    for var_expr in var_exprs:
        tavar = _create_analysis_var(scope,var_expr)
        tavars.append(ivar2)
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
                tavar = _create_analysis_var(scope,expr)
                if tavar["rank"] > 0:
                    raise util.error.LimitationError("reduction of arrays or array members not supported")
                tavar = copy.deepcopy(tavar)
                tavar["op"] = reduction_op
                taglobal_reduced_vars.append(tavar)
            try:
                all_vars2.remove(var_expr)
            except:
                pass # TODO error

    for var_expr in all_vars2:
        ivar = indexer.scope.search_scope_for_var(scope, var_expr)
        taglobal_vars.append(ivar)

    return taglobal_vars, taglobal_reduced_vars, tashared_vars, talocal_vars

def lookup_index_entries_for_vars_in_loopnest(scope,ttloopnest):
    return lookup_index_entries_for_vars_in_kernel_body(scope,
                                                        ttloopnest.vars_in_body(),
                                                        ttloopnest.gang_team_reductions(),
                                                        ttloopnest.gang_team_shared_vars(),
                                                        ttloopnest.local_scalars(),
                                                        ttloopnest.loop_vars())

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
    all_var_exprs = ttprocedure.vars_in_body() # in the body, there might be variables present from used modules
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

# OpenACC specific

def kernel_args_to_acc_mappings_no_types(acc_clauses,tavars,present_by_default,callback,**kwargs):
    r"""Derive mappings for arrays

    :return: A dictionary that maps the 'expr' field of a translator analysis var ('tavar')
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
    
    :raise util.parsing.LimitationError: If a derived type must be mapped.
    :raise util.parsing.SyntaxError: If the OpenACC clauses could not be parsed. 
    """
    mappings = {}
    for tavar in tavars:
        if tavar["rank"] > 0:
            explicitly_mapped = False
            for clause in acc_clauses:
                kind1, args = clause
                kind = kind1.lower()
                for var_expr1 in args:
                    var_expr = var_expr1.lower()
                    var_tag = util.parsing.strip_array_indexing(var_expr)
                    if tavar["expr"] == var_tag:
                        if "%" in var_tag:
                            raise util.parsing.LimitationError("mapping of derived type members not supported (yet)")
                        else:
                            mappings[tavar["expr"]] = callback(kind,var_expr,**kwargs)
                            explicitly_mapped = True
                            break
                if explicitly_mapped: break
            if not explicitly_mapped and present_by_default:
                if "%" in tavar["expr"] or tavar["f_type"]=="type": # TODO refine
                    raise util.parsing.LimitationError("mapping of derived types and their members not supported (yet)")
                else:
                    mappings[tavar["expr"]] = callback("present_or_copy",tavar["expr"],**kwargs)
            elif not explicitly_mapped:
                return util.parsing.SyntaxError("no mapping specified for expression: {}".format(tavar["expr"]))
    return mappings 
