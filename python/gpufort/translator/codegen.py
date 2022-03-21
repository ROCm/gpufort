from gpufort import util

from . import opts
from . import tree
from . import prepostprocess
from . import analysis

def translate_procedure_body_to_hip_kernel_body(ttprocedurebody, scope, **kwargs):
    """
    :return: body of a procedure as C/C++ code.
    Non-empty result names will be propagated to
    all return statements.
    """
    # 0. Clarify types of function calls / tensor access that are not
    # members of a struct
    lrvalues = analysis.find_all_matching_exclude_directives(ttprocedurebody.body,
                                                             lambda ttnode: isinstance(ttnode,tree.IValue))
    tree.transformations.flag_tensors(lrvalues, scope)
    
    # 1. Propagate result variable name to return statements
    if len(ttprocedurebody.result_name):
        for expr in tree.find_all(ttprocedurebody.body, tree.TTReturn):
            expr._result_name = result_name
    c_body = tree.make_c_str(ttprocedurebody.body)

    if len(ttprocedurebody.result_name):
        c_body += "\nreturn " + result_name + ";"
    return prepostprocess.postprocess_c_snippet(c_body)

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

    # 0. Clarify types of function calls / tensor access that are not
    # members of a struct
    lrvalues = analysis.find_all_matching_exclude_directives(ttloopnest.body,
                                                             lambda ttnode: isinstance(ttnode,tree.IValue))
    tree.transformations.flag_tensors(lrvalues, scope)
    substitutions = {}
    if map_to_flat_arrays:
        loop_vars = ttloopnest.loop_vars()
        substitutions = tree.transformations.map_allocatable_pointer_derived_type_members_to_flat_arrays(lrvalues,loop_vars,scope)
    
    # TODO look up correct signature of called device functions from index
    # 1.1 Collapsing
    num_outer_loops_to_map = int(ttloopnest.parent_directive().num_collapse())
    if opts.loop_collapse_strategy == "grid" and num_outer_loops_to_map <= 3:
        dim = num_outer_loops_to_map
    else: # "collapse" or num_outer_loops_to_map > 3
        dim = 1
    tidx = "__gidx{dim}".format(dim=dim)
    # 1. unpack colon (":") expressions
    for expr in tree.find_all(ttloopnest.body[0], tree.TTStatement):
        if type(expr._statement[0]) is tree.TTAssignment:
            expr._statement[0] = expr._statement[
                0].convert_to_do_loop_nest_if_necessary()
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
    # 3. collapse and transform do-loops
    do_loops = tree.find_all(ttloopnest.body[0], tree.TTDo)
    if num_outer_loops_to_map == 1 or (opts.loop_collapse_strategy
                                       == "grid" and
                                       num_outer_loops_to_map <= 3):
        indices, conditions = tree.transformations.map_loopnest_to_grid(do_loops,num_outer_loops_to_map)
    else: # "collapse" or num_outer_loops_to_map > 3
        indices, conditions = tree.transformations.collapse_loopnest(do_loops)
    
    c_snippet = "{0}\n{2}if ({1}) {{\n{3}\n}}".format(\
        "".join(indices),"&&".join(conditions),reduction_preamble,tree.make_c_str(ttloopnest.body[0]))

    return prepostprocess.postprocess_c_snippet(c_snippet), substitutions

def translate_loopnest_to_omp(fortran_snippet, ttloopnest, inout_arrays_in_body, arrays_in_body):
    """
    :note: The string used for parsing was preprocessed. Hence
           we pass the original Fortran snippet here.
    """

    # TODO There is only one loop or loop-like expression
    # in a parallel loop.
    # There might me multiple loops or look-like expressions
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
