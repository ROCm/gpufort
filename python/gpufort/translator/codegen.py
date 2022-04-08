# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import textwrap

from gpufort import util

from . import opts
from . import tree
from . import prepostprocess
from . import analysis
from . import parser
from . import transformations

def _modify_array_expressions(ttnode,lrvalues,scope,**kwargs):
    """:return: If any array expressions have been converted to loops.
    """
    fortran_style_tensor_access,_ = util.kwargs.get_value("fortran_style_tensor_access",opts.fortran_style_tensor_access,**kwargs)
    
    loop_ctr = transformations.expand_all_array_expressions(ttnode, scope, fortran_style_tensor_access)
    
    # TODO pass Fortran style access option down here too
    transformations.flag_tensors(lrvalues, scope)

    return loop_ctr > 1

def translate_procedure_body_to_hip_kernel_body(ttprocedurebody, scope, **kwargs):
    """
    :return: body of a procedure as C/C++ code.
    Non-empty result names will be propagated to
    all return statements.
    """
    lrvalues = analysis.find_all_matching_exclude_directives(ttprocedurebody.body,
                                                             lambda ttnode: isinstance(ttnode,tree.IValue))
    loops_generated = _modify_array_expressions(ttprocedurebody, lrvalues, scope, **kwargs)
    if loops_generated: # tree was modified
        lrvalues = analysis.find_all_matching_exclude_directives(ttloopnest.body,
                                                                 lambda ttnode: isinstance(ttnode,tree.IValue))
    c_body = tree.make_c_str(ttprocedurebody.body)

    # Append return statement if this is a function
    if ttprocedurebody != None and len(ttprocedurebody.result_name):
        c_body += "\nreturn " + ttprocedurebody.result_name + ";"
    return prepostprocess.postprocess_c_snippet(c_body)

def _handle_reductions(ttloopnest,grid_dim):
    tidx = "__gidx{dim}".format(dim=grid_dim)
    # 2. Identify reduced variables
    for expr in tree.find_all(ttloopnest.body[0], tree.TTAssignment):
        for value in tree.find_all_matching(
                expr, lambda x: isinstance(x, tree.IValue)):
            if type(value._value) in [
                    tree.TTDerivedTypeMember, tree.TTIdentifier
            ]:
                for op, reduced_vars in ttloopnest.gang_team_reductions(
                ).items():
                    if value.name().lower() in [
                            el.lower() for el in reduced_vars
                    ]:
                        value._reduction_index = tidx
        # TODO identify what operation is performed on the highest level to
        # identify reduction op
    reduction_preamble = ""
    # 2.1. Add init preamble for reduced variables
    for kind, reduced_vars in ttloopnest.gang_team_reductions(
            tree.make_c_str).items():
        for var in reduced_vars:
            if opts.fortran_style_tensor_access:
                reduction_preamble += "reduce_op_{kind}::init({var}({tidx}));\n".format(
                    kind=kind, var=var, tidx=tidx)
            else:
                reduction_preamble += "reduce_op_{kind}::init({var}[{tidx}]);\n".format(
                    kind=kind, var=var, tidx=tidx)
    return reduction_preamble

def translate_loopnest_to_hip_kernel_body(ttloopnest, scope, **kwargs):
    r"""This routine generates an HIP/C kernel body.
    :param ttloopnest: A translator tree node describing a loopnest
    :param scope: A scope; see gpufort.indexer.scope
    :param \*\*kwargs: keyword arguments.
    
    :return: A HIP C++ snippet and a list of substitutions that have
             been performed to the variables found in the body.
    """
    loop_collapse_strategy,_ = util.kwargs.get_value("loop_collapse_strategy",opts.loop_collapse_strategy,**kwargs)
    map_to_flat_arrays,_     = util.kwargs.get_value("map_to_flat_arrays",opts.map_to_flat_arrays,**kwargs)
    map_to_flat_scalars,_     = util.kwargs.get_value("map_to_flat_scalars",opts.map_to_flat_scalars,**kwargs)
    fortran_style_tensor_access,_ = util.kwargs.get_value("fortran_style_tensor_access",opts.fortran_style_tensor_access,**kwargs)

    lrvalues = analysis.find_all_matching_exclude_directives(ttloopnest.body,
                                                             lambda ttnode: isinstance(ttnode,tree.IValue))
    loops_generated = _modify_array_expressions(ttloopnest,lrvalues,scope,**kwargs)
    if loops_generated: # tree was modified
        lrvalues = analysis.find_all_matching_exclude_directives(ttloopnest.body,
                                                                 lambda ttnode: isinstance(ttnode,tree.IValue))
 
    ttdos = analysis.perfectly_nested_do_loops_to_map(ttloopnest) 
    problem_size = analysis.problem_size(ttdos,**kwargs)
    loop_vars = analysis.loop_vars_in_loopnest(ttdos)
    substitutions = {}
    if map_to_flat_arrays:
        substitutions.update(transformations.map_allocatable_pointer_derived_type_members_to_flat_arrays(lrvalues,loop_vars,scope))
    if map_to_flat_scalars:
        substitutions.update(transformations.map_scalar_derived_type_members_to_flat_scalars(lrvalues,loop_vars,scope))
    
    num_loops_to_map = len(ttdos)
    if loop_collapse_strategy == "grid" and num_loops_to_map <= 3:
        grid_dim = num_loops_to_map
    else: # "collapse" or num_loops_to_map > 3
        grid_dim = 1
    
    reduction_preamble = _handle_reductions(ttloopnest,grid_dim)
    
    # collapse and transform do-loops
    if (num_loops_to_map <= 1 
       or (loop_collapse_strategy == "grid" 
          and num_loops_to_map <= 3)):
        if len(reduction_preamble):
            reduction_preamble += "\n"
        indices, conditions = transformations.map_loopnest_to_grid(ttdos)
        c_snippet = "{0}\n{2}if ({1}) {{\n{3}\n}}".format(\
        "".join(indices),"&&".join(conditions),reduction_preamble,tree.make_c_str(ttloopnest.body[0]))
    else: # collapse strategy or num_loops_to_map > 3
        if len(reduction_preamble):
            reduction_preamble += "\n"
        preamble, indices, conditions = transformations.collapse_loopnest(ttdos)
        c_snippet = "{2}{4}{0}if ({1}) {{\n{3}\n}}".format(
            "".join(indices),
            "&&".join(conditions),
            reduction_preamble,
            tree.make_c_str(ttloopnest.body[0]),
            "".join(preamble))

    return prepostprocess.postprocess_c_snippet(c_snippet), problem_size, loop_vars, substitutions

def translate_loopnest_to_omp(fortran_snippet, ttloopnest, inout_arrays_in_body, arrays_in_body):
    """
    :note: The string used for parsing was preprocessed. Hence
           we pass the original Fortran snippet here.
    """

    # TODO There is only one loop or loop-like expression
    # in a parallel loop.
    # There might me multiple loops or loop-like expressions
    # in a kernels region.
    # kernels directives must be split
    # into multiple clauses.
    # In all cases the begin and end directives must
    # be consumed.
    # TODO find out relevant directives
    # TODO transform string
    # TODO preprocess Fortran colon expressions
    reduction = ttloopnest.gang_team_reductions()
    depend = ttloopnest.depend()
    if isinstance(ttloopnest.parent_directive(), tree.TTCufKernelDo):

        def cuf_kernel_do_repl(parse_result):
            nonlocal arrays_in_body
            nonlocal inout_arrays_in_body
            nonlocal reduction
            return parse_result.omp_f_str(arrays_in_body,
                                          inout_arrays_in_body, reduction,
                                          depend), True

        result,_ = util.pyparsing.replace_first(fortran_snippet,\
            tree.grammar.cuf_kernel_do,\
            cuf_kernel_do_repl)
        return result
    else:

        def acc_compute_repl(parse_result):
            nonlocal arrays_in_body
            nonlocal inout_arrays_in_body
            nonlocal reduction
            return parse_result.omp_f_str(arrays_in_body,
                                          inout_arrays_in_body,
                                          depend), True

        parallel_region = "parallel"

        def acc_loop_repl(parse_result):
            nonlocal arrays_in_body
            nonlocal inout_arrays_in_body
            nonlocal reduction
            nonlocal parallel_region
            result = parse_result.omp_f_str("do", parallel_region)
            parallel_region = ""
            return result, True

        def acc_end_repl(parse_result):
            nonlocal arrays_in_body
            nonlocal inout_arrays_in_body
            nonlocal reduction
            return parse_result.strip() + "!$omp end target", True

        result,_ = util.pyparsing.replace_first(fortran_snippet,\
                tree.grammar.acc_parallel | tree.grammar.acc_parallel_loop | tree.grammar.acc_kernels | tree.grammar.acc_kernels_loop,\
                acc_compute_repl)
        result,_ = util.pyparsing.replace_all(result,\
                tree.grammar.acc_loop,\
                acc_loop_repl)
        result,_ = util.pyparsing.replace_first(result,\
                tree.grammar.Optional(tree.grammar.White(),default="") + ( tree.grammar.ACC_END_PARALLEL | tree.grammar.ACC_END_KERNELS ),
                acc_end_repl)
        result,_ = util.pyparsing.erase_all(result,\
                tree.grammar.ACC_END_PARALLEL_LOOP | tree.grammar.ACC_END_KERNELS_LOOP)
        return result
