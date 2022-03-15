# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import textwrap

from gpufort import translator
from gpufort import util
from gpufort import indexer

from ... import opts

from .. import nodes
from .. import backends

from . import cufnodes

backends.supported_destination_dialects.add("hip")
dest_dialects = ["hip","hipgcc","hipgpufort"]

def CufLoopNest2Hip(stloopnest,*args,**kwargs):
    # map the parameters
    # TODO check for derived types
    for tavar in stloopnest.kernel_args_tavars:
        if tavar["rank"] > 0:
            var_expr = tavar["expr"]
            tokens = [
              "gpufort_array",tavar["rank"],"_wrap_device_cptr(&\n",
              " "*4,var_expr,",lbounds(",var_expr,"))",
            ]
            stloopnest.kernel_args_names.append("".join(tokens))
        else:
            stloopnest.kernel_args_names.append(tavar["expr"])
    return nodes.STLoopNest.transform(stloopnest,*args,**kwargs)

# backends for standard nodes
def handle_allocate_cuf(stallocate, joined_statements, index):
    indent = stallocate.first_line_indent()
    # CUF
    transformed = False
    bytes_per_element = []
    array_qualifiers = []
    for array_name in stallocate.parse_result.variable_names():
        ivar  = indexer.scope.search_index_for_var(index,stallocate.parent.tag(),\
          array_name)
        bytes_per_element.append(ivar["bytes_per_element"])
        qualifier, transformation_required = nodes.pinned_or_on_device(ivar)
        transformed |= transformation_required
        array_qualifiers.append(qualifier)
    subst = stallocate.parse_result.hip_f_str(bytes_per_element,
                                              array_qualifiers).lstrip(" ")
    return (textwrap.indent(subst,indent), transformed)


def handle_deallocate_cuf(stdeallocate, joined_statements, index):
    indent = stdeallocate.first_line_indent()
    transformed = False
    array_qualifiers = []
    for array_name in stdeallocate.parse_result.variable_names():
        ivar  = indexer.scope.search_index_for_var(index,stdeallocate.parent.tag(),\
          array_name)
        on_device = nodes.index_var_is_on_device(ivar)
        qualifier, transformed1 = nodes.pinned_or_on_device(ivar)
        transformed |= transformed1
        array_qualifiers.append(qualifier)
    subst = stdeallocate.parse_result.hip_f_str(array_qualifiers).lstrip(" ")
    return (textwrap.indent(subst,indent), transformed)

_cuf_2_hipfort = {
  "cudafor" : "hipfort",
  "cublas" : "hipblas",
  "cusparse" : "hipsparse",
  "cufft" : "hipfft",
  "curand" : "hipblas",
}

def handle_use_statement_cuf(stuse, joined_statements, index):
    """Removes CUDA Fortran use statements and 
    adds hipfort use statements instead. 
    """
    mod, only = util.parsing.parse_use_statement(joined_statements)

    # TODO handle only part
    cuf_module = mod in _cuf_2_hipfort
    if cuf_module:
        assert isinstance(stuse.parent,nodes.STContainerBase)
        stuse.parent.add_use_statement(_cuf_2_hipfort[mod])
        stuse.parent.add_use_statement("hipfort_check")
        stuse.parent.add_use_statement("iso_c_binding")
    return "", cuf_module

def handle_declaration_cuf(stdeclaration, joined_statements, index=[]):
    """
    if device and allocatable, remove device, add pointer
    if device and fixed size array, remove device, add pointer, replace fixed bounds by other bounds
       find first code line and add allocation to preamble if no dummy argument. Do similar things
       with deallocation -> double pass
    if pinned and allocatable, add pointer
    if pinned and fixed size array, remove pinned, add pointer, replace fixed bounds by other bounds
       find first code line and add allocation to preamble if no dummy argument -> double pass
    if integer with stream kind, 
    """
    stmt = joined_statements
    if len(index):
        index = index
    else:
        index = copy.copy(indexer.scope.EMPTY)
        variables = create_index_records_from_declaration(stmt.lower())
        index["variables"] += variables

    _, _, _, dimension_bounds, variables, original_datatype, original_qualifiers = util.parsing.parse_declaration(stmt)

    # argument names if declared in procedure
    if isinstance(stdeclaration.parent, nodes.STProcedure):
        argnames = list(stdeclaration.parent.index_record["dummy_args"])
    else:
        argnames = []

    result = []
    modified = False
    for var in variables:
        var_name, var_bounds, var_rhs = var
        ivar = indexer.scope.search_index_for_var(\
          index,stdeclaration.parent.tag(),\
            var_name)
        rank = ivar["rank"]
        has_device = "device" in ivar["qualifiers"]
        has_pinned = "pinned" in ivar["qualifiers"]
        has_pointer = "pointer" in ivar["qualifiers"]
        is_fixed_size_array = (rank > 0
                              and "allocatable" not in ivar["qualifiers"]
                              and not has_pointer)
        # 
        if has_device or has_pinned:
            # new qualifiers
            new_qualifiers = []
            for q in original_qualifiers:
                q_lower = q.lower()
                if (not q_lower in ["pointer","target", "pinned", "device", "allocatable"] 
                   and not q_lower.startswith("dimension")):
                    new_qualifiers.append(q)
            if var_name in argnames and not has_pointer:
                new_qualifiers.append("target")
            else:
                new_qualifiers.append("pointer")
            if rank > 0:
                new_qualifiers.append("dimension(:" + ",:" * (rank-1) + ")")
            # fixed size arrays
            if is_fixed_size_array:
                if isinstance(stdeclaration.parent,nodes.STModule):
                    return util.error.LimitationError("device array without pointer or allocatable qualifier not supported in module")
                malloc_tokens = ["call hipCheck(","hipMalloc","(",var_name,",",",".join(var_bounds+dimension_bounds),"))"]
                free_tokens   = ["call hipCheck(","hipFree","(",var_name,"))"]
                if has_pinned: 
                    malloc_tokens[1] = "hipHostMalloc" 
                    free_tokens[1] = "hipHostFree"
                    flags = "0"
                    malloc_tokens.insert(-1,",")
                    malloc_tokens.insert(-1,flags)
                stdeclaration.parent.append_to_decl_list(["".join(malloc_tokens)])
                stdeclaration.parent.prepend_to_return_or_end_statements(["".join(free_tokens)])
            # modified variable declaration
            tokens = [original_datatype]
            if len(new_qualifiers):
                tokens += [",",",".join(new_qualifiers)]
            result.append("".join(tokens+[" :: ",var_name]))
            modified = True
        else:
            tokens = [original_datatype]
            if len(original_qualifiers):
                tokens += [",",",".join(original_qualifiers)]
            tokens += [" :: ",var_name]
            if len(var_bounds):
                tokens.append("({})".format(",".join(var_bounds)))
            if var_rhs != None:
                tokens.append(" => " if has_pointer else " = ")
                tokens.append(var_rhs)
            result.append("".join(tokens))
        indent = stdeclaration.first_line_indent()
        return textwrap.indent("\n".join(result),indent), modified

# post processing
def _handle_cublas_v1(stree):
    """If a cublas v1 call is present in a program/module/procedure,
    add hipblas handle to the declaration list and a hipblasCreate/-Destroy
    call before the first cublas call and after the last one, respectively.
    """
    # cublas_v1 detection
    if opts.cublas_version == 1:

        def has_cublas_call_(child):
            return type(child) is cufnodes.STCufLibCall and child.has_cublas()

        cuf_cublas_calls = stree.find_all(filter=has_cublas_call_,
                                          recursively=True)
        for call in cuf_cublas_calls:
            last_decl_list_node = call.parent.last_entry_in_decl_list()
            indent = last_decl_list_node.first_line_indent()
            last_decl_list_node.add_to_epilog(
                "{0}type(c_ptr) :: hipblasHandle = c_null_ptr\n".format(
                    indent))

            local_cublas_calls = call.parent.find_all(filter=has_cublas_call,
                                                      recursively=False)
            first = local_cublas_calls[0]
            indent = first.first_line_indent()
            first.add_to_prolog(
                "{0}hipblasCreate(hipblasHandle)\n".format(indent))
            last = local_cublas_calls[-1]
            indent = last.first_line_indent()
            last.add_to_epilog(
                "{0}hipblasDestroy(hipblasHandle)\n".format(indent))


@util.logging.log_entry_and_exit(opts.log_prefix)
def postprocess_tree_cuf(stree, index):
    _handle_cublas_v1(stree)

cufnodes.STCufLoopNest.register_backend(dest_dialects,CufLoopNest2Hip)

nodes.STAllocate.register_backend("cuf", dest_dialects, handle_allocate_cuf)
nodes.STDeallocate.register_backend("cuf", dest_dialects, handle_deallocate_cuf)
nodes.STUseStatement.register_backend("cuf", dest_dialects, handle_use_statement_cuf)
nodes.STDeclaration.register_backend("cuf", dest_dialects, handle_declaration_cuf)

backends.register_postprocess_backend("cuf", dest_dialects, postprocess_tree_cuf)
