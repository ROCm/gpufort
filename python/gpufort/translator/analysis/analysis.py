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

from .. import tree
from .. import conv
from .. import opts

def append_c_type(tavar):
    f_type = tavar["f_type"]
    kind = tavar["kind"]
    if f_type == "type":
        tavar["c_type"] = tavar["kind"] 
    elif f_type == "character":
        tavar["c_type"] = "char"
        # todo: more carefully check if len or kind is specified for characters
    elif f_type != "character": 
        tavar["c_type"] = conv.convert_to_c_type(f_type, kind, "TODO unknown")
    tavar["bytes_per_element"] = conv.num_bytes(f_type, kind, default=None)

def create_analysis_var(scope, var_expr):
    ivar = indexer.scope.search_scope_for_var(
        scope, var_expr)
    tavar = copy.deepcopy(ivar)
    tavar["expr"] = util.parsing.strip_array_indexing(var_expr).lower()
    tavar["c_name"] = tavar["name"] 
    tavar["c_rank"] = tavar["rank"]
    tavar["op"]   = ""
    append_c_type(tavar)
    if tavar["rank"] > 0:
        tavar["lbounds"]    = []
        tavar["ubounds"]    = []
        tavar["size"]       = []
        unknown_size = False
        for arg in tavar["bounds"]:
            lbound,ubound,stride,size = None,None,None,None
            tokens = util.parsing.tokenize(arg)
            parts,_ = util.parsing.get_top_level_operands(tokens,
                        separators=[":"])
            if len(parts) == 0:
                lbound = None
                ubound = None
                stride = None
                size = None
                unknown_size = True
            elif len(parts) == 1:
                lbound = "1"
                ubound = parts[0]
                stride = "1"
                size = ubound
            elif len(parts) == 2:
                lbound = parts[0]
                ubound = parts[1]
                stride = "1"
                size = "1+(({}) - ({}))".format(ubound,lbound)
            elif len(parts) == 3:
                lbound = parts[0]
                ubound = parts[1]
                stride = parts[2]
                if stride != "1":
                    raise util.error.LimitationError("cannot map arrays with stride other than '1'")
                size = "1+(({}) - ({}))".format(ubound,lbound)
            else:
                raise util.error.SyntaxError("expected 1,2,or 3 colon-separated expressions")
            tavar["lbounds"].append(lbound)
            tavar["ubounds"].append(ubound)
            tavar["size"].append(size)
        if unknown_size:
            tavar["c_size_bytes"] = None
        else:
            tavar["c_size_bytes"] = "sizeof({})*{}".format(tavar["c_type"],"*".join(tavar["size"]))
    return tavar

def _lookup_index_vars(scope, var_exprs, consumed_var_exprs=[]):
    """Search scope for index vars and remove corresponding 
       var expression from all_vars2 list."""
    tavars = []
    for var_expr in var_exprs:
        tavar = create_analysis_var(scope,var_expr)
        if not "parameter" in tavar["attributes"] or tavar["rank"] > 0:
            tavars.append(tavar)
        consumed_var_exprs.append(var_expr)
    return tavars

def lookup_index_entries_for_vars_in_kernel_body(scope,
                                                 all_vars,
                                                 reductions,
                                                 shared_vars,
                                                 local_vars,
                                                 loop_vars):
    """Lookup index variables
    :param list all_vars: List of all variable expressions (var)
    :param list reductions: List of tuples pairing a reduction operation with the associated
                             variable expressions
    :param list shared:     List of variable expressions that are shared by the workitems/threads in a workgroup/threadblock
    :param list local_vars: List of variable expressions that can be mapped to local variables per workitem/thread
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
                tavar1 = create_analysis_var(scope,var_expr)
                if tavar1["rank"] > 0:
                    raise util.error.LimitationError("reduction of arrays or array members not supported")
                elif "parameter" in tavar1["attributes"]:
                    raise util.error.SyntaxError("parameters cannot be reduced")
                tavar = copy.deepcopy(tavar1)
                tavar["op"] = conv.get_operator_name(reduction_op)
                taglobal_reduced_vars.append(tavar)
            try:
                all_vars2.remove(var_expr)
            except:
                pass # todo: error

    for var_expr in all_vars2:
        tavar = create_analysis_var(scope, var_expr)
        if not "parameter" in tavar["attributes"] or tavar["rank"] > 0:
            taglobal_vars.append(tavar)
    return taglobal_vars, taglobal_reduced_vars, tashared_vars, talocal_vars

def _apply_c_names(tavars,c_names):
    for i,tavar in enumerate(tavars):
        var_expr = tavar["expr"]
        if var_expr in c_names:
            tavar["c_name"] = c_names[var_expr]

def _apply_c_ranks(tavars,c_ranks):
    for i,tavar in enumerate(tavars):
        var_expr = tavar["expr"]
        if var_expr in c_ranks:
            tavar["c_rank"] = c_ranks[var_expr]

def lookup_index_entries_for_vars_in_compute_construct(scope,ttcomputeconstruct,loop_vars,c_names={},c_ranks={}):
    all_vars       = vars_in_subtree(ttcomputeconstruct, scope)
    local_scalars_ = local_scalars(ttcomputeconstruct,scope)
    local_vars     = [v for v in (local_scalars_+ttcomputeconstruct.gang_private_vars())
                     if v not in loop_vars]
    local_vars    += [v for v in all_vars 
                     if (v[0]=="_" 
                        and v not in loop_vars 
                        and v not in local_vars)]
    # todo: improve
    result = lookup_index_entries_for_vars_in_kernel_body(scope,
                                                          all_vars,
                                                          ttcomputeconstruct.gang_reductions(),
                                                          ttcomputeconstruct.gang_shared_vars(),
                                                          local_vars,
                                                          loop_vars)
    for i in range(0,4):
        _apply_c_names(result[i],c_names)
        _apply_c_ranks(result[i],c_ranks)
    return result

def lookup_index_entries_for_vars_in_procedure_body(scope,ttprocedurebody,iprocedure):
    shared_vars = [
        ivar["name"]
        for ivar in iprocedure["variables"]
        if "shared" in ivar["attributes"]
    ]
    all_var_exprs = vars_in_subtree(ttprocedurebody, scope) # in the body, there might be variables present from used modules
    local_vars = [
        v for v in all_var_exprs
        if (v not in iprocedure["dummy_args"]
           and v not in shared_vars
           and v not in tree.grammar.DEVICE_PREDEFINED_VARIABLES) # todo: only with
                                                                             # CUDA Fortran
    ]
    all_vars = iprocedure["dummy_args"] + shared_vars + local_vars
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
    """
    return (taglobal_vars
            + taglobal_reduced_vars)
            #+ [tavar for tavar in tashared_vars if tavar["rank"] > 0]
            #+ [tavar for tavar in talocal_vars if tavar["rank"] > 0])
    
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
            for el in curr.child_nodes():
                traverse_(el)
    traverse_(ttnode)
    return result
def search_value_exprs_in_subtree(ttnode, search_filter, scope, min_rank=-1):
    """All lvalue and ralue exprs in the subtree.
    :note: Returns simplified expressions, stripped off all array indexing syntax.
    :note: Excludes device predefined variables such as `threadidx%x` etc.
    """
    tags = []
    for ttvalue in find_all_matching_exclude_directives(
            ttnode.body,
            search_filter): # includes the identifiers of the function calls 
        # todo: Search tag creation strips array brackets
        # todo: provide clean interface to value
        tag = indexer.scope.create_index_search_tag(ttvalue._value.fstr()) # todo: 
        if tag not in tree.grammar.DEVICE_PREDEFINED_VARIABLES:
            ivar = indexer.scope.search_scope_for_var(scope, tag)
            if ivar["rank"] >= min_rank and\
               not tag in tags: # ordering important
                tags.append(tag)
    return tags


def vars_in_subtree(ttnode, scope):
    """:return: all identifiers of LValue and RValues in the body."""

    def search_filter(node):
        cond1 = (isinstance(node,tree.TTValue)
                and isinstance(node._value, (tree.TTDerivedTypePart,tree.TTIdentifier,tree.TTFunctionCall)))
        if cond1 and isinstance(node._value, tree.TTFunctionCall):
            return node._value.is_array_expr
        else:
            return cond1 
    result = search_value_exprs_in_subtree(ttnode, search_filter, scope)
    return result

def arrays_in_subtree(ttnode, scope):

    def search_filter(node):
        cond1 = (isinstance(node,tree.TTValue) 
                and isinstance(node._value, tree.TTFunctionCall))
        if cond1 and isinstance(node._value, tree.TTFunctionCall):
            return node._value.is_array_expr
        else:
            return cond1 

    return search_value_exprs_in_subtree(ttnode, search_filter, scope, 1)


def inout_arrays_in_subtree(ttnode, scope):

    def search_filter(node):
        cond1 = (isinstance(node,tree.TTLvalue) 
                and isinstance(node._value, tree.TTFunctionCall))
        if cond1 and isinstance(node._value, tree.TTFunctionCall):
            return node._value.is_array_expr
        else:
            return cond1 

    return search_value_exprs_in_subtree(ttnode, search_filter, scope, 1)

#def all_unmapped_arrays(ttcomputeconstruct, scope):
#    """:return: Name of all unmapped array variables"""
#    mapped_vars = ttcomputeconstruct.all_mapped_vars()
#    if ttcomputeconstruct.all_arrays_are_on_device():
#        return []
#    else:
#        return [var for var in arrays_in_body if not var in mapped_vars]
#
#def deviceptrs(ttcomputeconstruct):
#    if ttcomputeconstruct.all_arrays_are_on_device():
#        return arrays_in_subtree(ttcomputeconstruct, scope)
#    else:
#        return ttcomputeconstruct.deviceptrs()
    
def local_scalars_and_reduction_candidates(ttcomputeconstruct, scope):
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
        ttcomputeconstruct.body[0], lambda node: isinstance(node,(
            tree.TTAssignment, tree.TTComplexAssignment, tree.
            TTMatrixAssignment
        )))
    for assignment in assignments:
        # lhs scalars
        lvalue = assignment._lhs
        lvalue_name = lvalue._value.fstr().lower()
        if isinstance(lvalue, tree.TTIdentifier): # could still be a matrix
            ivar = indexer.scope.search_scope_for_var(
                scope, lvalue_name)
            if ivar["rank"] == 0 and\
               not lvalue_name in scalars_read_so_far:
                initialized_scalars.append(
                    lvalue_name) # read and initialized in
        # rhs scalars
        ttrvalues = tree.find_all(assignment._rhs,tree.TTRvalue)
        for ttrvalue in ttrvalues:
            if isinstance(ttrvalue._value,tree.TTIdentifier):
                rvalue_name = ttrvalue._value.fstr().lower()
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

def local_scalars(ttcomputeconstruct, scope):
    local_scalars, _ = local_scalars_and_reduction_candidates(ttcomputeconstruct, scope)
    return local_scalars

def reduction_candidates(ttcomputeconstruct, scope):
    _, reduction_candidates = local_scalars_and_reduction_candidates(ttcomputeconstruct, scope)
    return reduction_candidates

def loop_vars_in_compute_construct(ttdos):
    identifier_names = []
    for ttdo in ttdos:
        identifier_names.append(ttdo.loop_var(tree.make_fstr))
    return identifier_names

def _perfectly_nested_outer_do_loops(ttcomputeconstruct):
    """:return: Perfectly nested do loops, comments are ignored. 
    """
    ttdos = []
    def descend_(ttcontainer,depth=1):
        nonlocal ttdos
        for child in ttcontainer:
            if isinstance(child,tree.TTDo):
                ttdos.append(child)
                descend_(child,depth+1)
            elif not isinstance(child,(tree.TTCommentedOut,tree.TTCycle,tree.TTLabel)):
                # if there are other statements on the same level
                # and a do loop was added, remove it and its
                # inner loops. Then break the loop.
                while len(ttdos) >= depth:
                    ttdos.pop(-1)
                break
    descend_(ttcomputeconstruct)
    return ttdos

def perfectly_nested_do_loops_to_map(ttcomputeconstruct):
    num_loops_to_map = int(ttcomputeconstruct.parent_directive().num_collapse())
    ttdos = _perfectly_nested_outer_do_loops(ttcomputeconstruct)
    if (num_loops_to_map <= 0
       or num_loops_to_map > len(ttdos)):
        return ttdos
    else:
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
    
    :param list acc_clauses:        The acc clauses; list of tuples with clause kind as first and list of arguments as second argument.
    :param list tavars:             list of translator analysis vars
    :param bool present_by_default: If array variables without explicit mappings are present
                                    by default.
    :param callable callback:       A callback that takes the OpenACC clause as first
                                    argument, the expression subject to the mapping as second, 
                                    and the (implementation-specific) keyword arguments as third argument. 
  
    :param \*\*kwargs: Keyword args to pass to the callback.
    
    :note: Current implementation does not support mapping of scalars/arrays of derived type.
    :note: Clause list is processed in reversed order as the first clauses might originate from preceding
           data directives.
    :raise util.error.LimitationError: If a derived type must be mapped.
    :raise util.error.SyntaxError: If the OpenACC clauses could not be parsed. 
    """
    mappings = []
    for tavar in tavars:
        if tavar["rank"] > 0: # map array
            explicitly_mapped = False
            for clause in reversed(acc_clauses): # finds first matching clause from end of list on
                kind1, args = clause
                kind = kind1.lower()
                for var_expr in args:
                    var_tag = util.parsing.strip_array_indexing(var_expr.lower())
                    if tavar["expr"] == var_tag:
                        if tavar["f_type"]=="type":
                            raise util.error.LimitationError("mapping of derived type scalars and arrays not supported (yet)")
                        mappings.append((var_expr, callback(kind,var_expr,tavar,**kwargs)))
                        explicitly_mapped = True
                        break
                if explicitly_mapped: break
            if not explicitly_mapped and present_by_default:
                if tavar["f_type"]=="type": # todo: refine
                    raise util.error.LimitationError("mapping of derived type scalars and arrays not supported (yet)")
                else:
                    mappings.append((tavar["expr"], callback("copy",tavar["expr"],tavar,**kwargs)))
            elif not explicitly_mapped:
                raise util.error.SyntaxError("no mapping specified for expression: {}".format(tavar["expr"]))
        elif tavar["f_type"] == "type": # map type
            raise util.error.LimitationError("mapping of derived types and their members not supported (yet)")
        else: # basic type scalars
            mappings.append((tavar["expr"],tavar["expr"]))
    return mappings 
