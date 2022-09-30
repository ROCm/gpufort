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

def _modify_array_expressions(ttnode,ttvalues,scope,**kwargs):
    """:return: If any array expressions have been converted to loops.
    """
    fortran_style_tensor_access,_ = util.kwargs.get_value("fortran_style_tensor_access",opts.fortran_style_tensor_access,**kwargs)

    loop_ctr = transformations.expand_all_array_expressions(ttnode, scope, fortran_style_tensor_access)
    
    # todo: pass Fortran style access option down here too
    transformations.flag_tensors(ttvalues, scope)

    return loop_ctr > 1

def translate_procedure_body_to_hip_kernel_body(ttprocedurebody, scope, **kwargs):
    """
    :return: body of a procedure as C/C++ code.
    Non-empty result names will be propagated to
    all return statements.
    """
    ttvalues = analysis.find_all_matching_exclude_directives(
            ttprocedurebody.body,lambda ttnode: isinstance(ttnode,tree.TTValue))
    _modify_array_expressions(ttprocedurebody, ttvalues, scope, **kwargs)
    ttvalues = analysis.find_all_matching_exclude_directives(
            ttprocedurebody.body,(lambda ttnode: isinstance(ttnode,tree.TTValue) 
            and not ttnode.is_function_call()))
    
    c_body = tree.make_cstr(ttprocedurebody.body)

    # Append return statement if this is a function
    if (ttprocedurebody.result_name != None
       and len(ttprocedurebody.result_name)):
        c_body += "\nreturn " + ttprocedurebody.result_name + ";"
    return prepostprocess.postprocess_c_snippet(c_body)

def _handle_reductions(ttcomputeconstruct,ttvalues,grid_dim):
    tidx = "__gidx{dim}".format(dim=grid_dim)
    # 2. Identify reduced variables
    for ttvalue in ttvalues:
        if type(ttvalue._value) in [
                tree.TTDerivedTypeMember, tree.TTIdentifier
        ]:
            for op, reduced_vars in ttcomputeconstruct.gang_reductions(
            ).items():
                if ttvalue.name().lower() in [
                        el.lower() for el in reduced_vars
                ]:
                    ttvalue._reduction_index = tidx
    # todo: identify what operation is performed on the highest level to
    # identify reduction op
    reduction_preamble = ""
    # 2.1. Add init preamble for reduced variables
    for kind, reduced_vars in ttcomputeconstruct.gang_reductions(
            tree.make_cstr).items():
        for var in reduced_vars:
            if opts.fortran_style_tensor_access:
                reduction_preamble += "reduce_op_{kind}::init({var}({tidx}));\n".format(
                    kind=kind, var=var, tidx=tidx)
            else:
                reduction_preamble += "reduce_op_{kind}::init({var}[{tidx}]);\n".format(
                    kind=kind, var=var, tidx=tidx)
    return reduction_preamble

def translate_compute_construct_to_hip_kernel_body(ttcomputeconstruct, scope, **kwargs):
    r"""This routine generates an HIP/C kernel body.
    :param ttcomputeconstruct: A translator tree node describing a loopnest
    :param scope: A scope; see gpufort.indexer.scope
    :param \*\*kwargs: keyword arguments.
    
    :return: A HIP C++ snippet and a list of c_names that have
             been performed to the variables found in the body.
    """
    map_to_flat_arrays,_     = util.kwargs.get_value("map_to_flat_arrays",opts.map_to_flat_arrays,**kwargs)
    map_to_flat_scalars,_     = util.kwargs.get_value("map_to_flat_scalars",opts.map_to_flat_scalars,**kwargs)
    fortran_style_tensor_access,_ = util.kwargs.get_value("fortran_style_tensor_access",opts.fortran_style_tensor_access,**kwargs)

    ttvalues = analysis.find_all_matching_exclude_directives(ttcomputeconstruct.body,
                                                             lambda ttnode: isinstance(ttnode,tree.TTValue))
    _modify_array_expressions(ttcomputeconstruct,ttvalues,scope,**kwargs)
    ttvalues = analysis.find_all_matching_exclude_directives(
            ttcomputeconstruct.body,(lambda ttnode: isinstance(ttnode,tree.TTValue) 
            and not ttnode.is_function_call()))
    #print(ttvalues)
    c_ranks = transformations.adjust_explicitly_mapped_arrays_in_rank(ttvalues,ttcomputeconstruct.all_mapped_vars())
    # todo: Investigate what happens if such an array is mapped to flat array

    if ttcomputeconstruct.is_serial_construct():
        ttdos        = []
        problem_size = ["1"]
        block_size   = ["1"]
        loop_vars    = []
    else:
        ttdos = analysis.perfectly_nested_do_loops_to_map(ttcomputeconstruct) 
        problem_size = analysis.problem_size(ttdos,**kwargs)
        block_size = [] # todo:
        loop_vars = analysis.loop_vars_in_compute_construct(ttdos)

    c_names = {}
    if map_to_flat_arrays:
        c_names.update(transformations.map_allocatable_pointer_derived_type_members_to_flat_arrays(ttvalues,loop_vars,scope))
    if map_to_flat_scalars:
        c_names.update(transformations.map_scalar_derived_type_members_to_flat_scalars(ttvalues,loop_vars,scope))
    
    num_loops_to_map = len(ttdos)
    if (loop_collapse_strategy == "grid" 
       and num_loops_to_map <= 3
       and num_loops_to_map > 0):
        grid_dim = num_loops_to_map
    else: # "collapse" or num_loops_to_map > 3
        grid_dim = 1
    
    reduction_preamble = _handle_reductions(ttcomputeconstruct,ttvalues,grid_dim)
    
    # collapse and transform do-loops
    if ttcomputeconstruct.is_serial_construct(): 
        if len(reduction_preamble):
            reduction_preamble += "\n"
        c_snippet = "{0}if ( threadIdx.x==0 ) {{\n{1}\n}}".format(reduction_preamble,tree.make_cstr(ttcomputeconstruct))
    else:
       # todo: work with transformation result
       # todo: collapse CUDA Fortran loops by default
       if len(reduction_preamble):
           reduction_preamble += "\n"
       preamble, indices, conditions = transformations.collapse_loopnest(ttdos)
       c_snippet = "{2}{4}{0}if ({1}) {{\n{3}\n}}".format(
           "".join(indices),
           "&&".join(conditions),
           reduction_preamble,
           tree.make_cstr(ttcomputeconstruct),
           "".join(preamble))

    return prepostprocess.postprocess_c_snippet(c_snippet), problem_size, block_size, loop_vars, c_names, c_ranks

def translate_compute_construct_to_omp(fortran_snippet, ttcomputeconstruct, inout_arrays_in_body, arrays_in_body):
    """
    :note: The string used for parsing was preprocessed. Hence
           we pass the original Fortran snippet here.
    """

    # todo: There is only one loop or loop-like expression
    # in a parallel loop.
    # There might me multiple loops or loop-like expressions
    # in a kernels region.
    # kernels directives must be split
    # into multiple clauses.
    # In all cases the begin and end directives must
    # be consumed.
    # todo: find out relevant directives
    # todo: transform string
    # todo: preprocess Fortran colon expressions
    reduction = ttcomputeconstruct.gang_reductions()
    depend = ttcomputeconstruct.depend()
    if isinstance(ttcomputeconstruct.parent_directive(), tree.TTCufKernelDo):

        def cuf_kernel_do_repl(parse_result):
            nonlocal arrays_in_body
            nonlocal inout_arrays_in_body
            nonlocal reduction
            return parse_result.omp_fstr(arrays_in_body,
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
            return parse_result.omp_fstr(arrays_in_body,
                                          inout_arrays_in_body,
                                          depend), True

        parallel_region = "parallel"

        def acc_loop_repl(parse_result):
            nonlocal arrays_in_body
            nonlocal inout_arrays_in_body
            nonlocal reduction
            nonlocal parallel_region
            result = parse_result.omp_fstr("do", parallel_region)
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