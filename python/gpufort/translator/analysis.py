# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import indexer
from gpufort import grammar

def strip_member_access(var_exprs):
    """Strip off member access parts from derived type member access expressions,
       e.g. 'a%b%c' becomes 'a'.
    """
    # TODO only strip certain derived type expressions, e.g. if a struct is copied via copy(<struct>)
    # or if a CUDA Fortran derived type has the device attribute
    result = []
    return [var_expr.split("%", maxsplit=1)[0] for var_expr in var_exprs]

def _lookup_index_vars(scope, var_exprs, consumed_var_exprs=[]):
    """Search scope for index vars and remove corresponding 
       var expression from all_vars2 list."""
    ivars = []
    for var_expr in var_exprs:
        ivar = indexer.scope.search_scope_for_var(
            scope, var_expr)
        ivars.append(ivar)
        consumed_var_exprs.append(var_expr)
    return ivars

def lookup_index_entries_for_vars_in_kernel_body(scope,
                                                 all_vars,
                                                 reductions,
                                                 shared_vars,
                                                 local_vars,
                                                 loop_vars,
):
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
    # TODO parse bounds and right-hand side expressions here too

    consumed = []
    ilocal_vars = _lookup_index_vars(
        scope,
        strip_member_access(local_vars), consumed)
    ishared_vars = _lookup_index_vars(
        scope,
        strip_member_access(shared_vars), consumed)
    all_vars2 = [
        v for v in strip_member_access(all_vars)
        if not v in consumed and v not in loop_vars
    ]

    rglobal_reduced_vars = []
    iglobal_vars = []

    for reduction_op, var_exprs in reductions.items():
        for var_expr in var_exprs:
            if "%" in var_expr:
                raise util.error.LimitationError("reduction of derived type members not supported")
            else:
                ivar = indexer.scope.search_scope_for_var(
                    scope, var_expr)
                if ivar["rank"] > 0:
                    raise util.error.LimitationError("reduction of arrays or array members not supported")
                rvar = copy.deepcopy(ivar)
                rvar["op"] = reduction_op
                rglobal_reduced_vars.append(rvar)
            try:
                all_vars2.remove(var_expr)
            except:
                pass # TODO error

    for var_expr in all_vars2:
        ivar = indexer.scope.search_scope_for_var(scope, var_expr)
        iglobal_vars.append(ivar)

    return iglobal_vars, rglobal_reduced_vars, ishared_vars, ilocal_vars

def lookup_index_entries_for_vars_in_loopnest(scope,ttloopnest,):
    return lookup_index_entries_for_vars_in_kernel_body(scope,
                                                        ttloopnest.vars_in_body(),
                                                        ttloopnest.gang_team_reductions(),
                                                        ttloopnest.gang_team_shared_vars(),
                                                        ttloopnest.local_scalars(),
                                                        ttloopnest.loop_vars(),
                                                        )

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
    all_var_exprs = analysis.strip_member_access(ttprocedure.vars_in_body(
    )) # in the body, there might be variables present from used modules
    all_vars = iprocedure["dummy_args"] + [
        v for v in all_var_exprs if (v not in iprocedure["dummy_args"] and
                                     v not in grammar.DEVICE_PREDEFINED_VARIABLES)
    ]
    global_reductions = {}
    loop_vars = []

    return lookup_index_entries_for_vars_in_kernel_body(scope,
                                                        all_vars,
                                                        global_reductions,
                                                        shared_vars,
                                                        local_vars,
                                                        loop_vars,
                                                        )

def get_kernel_arguments(iglobal_vars, rglobal_reduced_vars, ishared_vars, ilocal_vars):
    """:return: index records for the variables
                that must be passed as kernel arguments.
    :note: Shared_vars and local_vars must be passed as kernel argument
           as well if the respective variable is an array. 
    """
    return (iglobal_vars
            + rglobal_reduced_vars
            + [ivar for ivar in ishared_vars if ivar["rank"] > 0]
            + [ivar for ivar in ilocal_vars if ivar["rank"] > 0])
