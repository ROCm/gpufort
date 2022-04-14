# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import copy
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
              "gpufort_array",str(tavar["rank"]),"_wrap_device_cptr(&\n",
              " "*4,"c_loc(",var_expr,"),shape(",var_expr,",kind=c_int),lbound(",var_expr,",kind=c_int))",
            ]
            stloopnest.kernel_args_names.append("".join(tokens))
        else:
            stloopnest.kernel_args_names.append(tavar["expr"])
    return nodes.STLoopNest.transform(stloopnest,*args,**kwargs)

# backends for standard nodes
def hip_f_str(self,
              bytes_per_element,
              array_qualifiers,
              vars_are_c_ptrs=False):
    """Generate HIP ISO C Fortran expression for all
    device and pinned host allocations.
    Use standard allocate for all other allocations.
    :param array_qualifiers: List storing per variable, one of 'managed', 'constant', 'shared', 'pinned', 'texture', 'device' or None.
    :see: variable_names(self) 
    """
    assert len(bytes_per_element) is len(self._vars)
    assert len(array_qualifiers) is len(self._vars)
    result = []
    other_arrays = []
    for i, array in enumerate(self._vars):
        if vars_are_c_ptrs:
            size = array.size(bytes_per_element[i],
                              base.make_f_str) # total size in bytes
        else:
            size = ",".join(
                array.counts_f_str()) # element counts per dimension
        if array_qualifiers[i] == "device":
            line = "call hipCheck(hipMalloc({0}, {1}))".format(
                array.var_name(), size)
            result.append(line)
        elif array_qualifiers[i] == "pinned":
            line = "call hipCheck(hipHostMalloc({0}, {1}, 0))".format(
                array.var_name(), size)
            result.append(line)
        else:
            other_arrays.append(base.make_f_str(array))
        if vars_are_c_ptrs and not array_qualifiers[i] in [
                "pinned", "device"
        ]:
            result += array.bound_var_assignments(array.var_name())
    if len(other_arrays):
        line = "allocate({0})".format(",".join(other_arrays))
        result.append(line)
    return "\n".join(result)

def _render_allocation_f_str(name,bounds):
    args = []
    for alloc_range in bounds:
        args.append(":".join([el for el in alloc_range if el != None]))
    return "".join([name,"(",",".join(args),")"])

def _render_allocation_hipfort_f_str(api,name,bounds,stat):
    counts  = []
    lbounds = []
    for alloc_range in bounds:
        lbound, ubound, stride = alloc_range
        if stride != None:
            try:
                step_as_int = int(stride)
                if step_as_int != 1:
                    raise error.LimitationError("stride != 1 not supported")
            except ValueError:
                    raise error.LimitationError("non-numeric stride not supported")
        # can assume stride = 1 below
        all_lbounds_are_one = True
        if lbound == None:
            counts.append(ubound)
            lbounds.append("1")
        else:
            all_lbounds_are_one = False
            counts.append("1+(({1})-({0}))".format(lbound,ubound))
            lbounds.append(lbound)
    if all_lbounds_are_one:
        args = copy.copy(counts)
    else:
        args = []
        args.append("".join(["[",",".join(counts),"]"]))
        args.append("".join(["lbounds=[",",".join(lbounds),"]"]))
    if stat != None:
        prefix = "".join([stat,"="])
        suffix = ""
    else:
        prefix = "call hipCheck("
        suffix = ")"
    return "".join([prefix,api,"(",name,",",",".join(args),")",suffix])

def handle_allocate_cuf(stallocate, joined_statements, index):
    indent = stallocate.first_line_indent()
    unchanged_allocation_args = []
    hipfort_allocations = []
    stat = stallocate.stat
    for name, bounds in stallocate.allocations:
        ivar  = indexer.scope.search_index_for_var(index,stallocate.parent.tag(),\
          name)
        if "pinned" in ivar["qualifiers"]:
            hipfort_allocations.append(_render_allocation_hipfort_f_str("hipHostMalloc",name,bounds,stat))
        elif "device" in ivar["qualifiers"]:
            hipfort_allocations.append(_render_allocation_hipfort_f_str("hipMalloc",name,bounds,stat))
        else:
            unchanged_allocation_args.append(_render_allocation_f_str(name,bounds))
    result = []
    if len(unchanged_allocation_args):
        if stat != None:
            unchanged_allocation_args.append("".join(["stat=",stat]))
        result.append("".join(["allocate","(",",".join(unchanged_allocation_args),")"]))
    result += hipfort_allocations
    return (textwrap.indent("\n".join(result),indent), len(hipfort_allocations))

def _render_deallocation_hipfort_f_str(api,name,stat):
    if stat != None:
        prefix = "".join([stat,"="])
        suffix = ""
    else:
        prefix = "call hipCheck("
        suffix = ")"
    return "".join([prefix,api,"(",name,")",suffix])

def handle_deallocate_cuf(stdeallocate, joined_statements, index):
    indent = stdeallocate.first_line_indent()
    unchanged_deallocation_args = []
    hipfort_deallocations = []
    stat = stdeallocate.stat
    for name in stdeallocate.variable_names:
        ivar  = indexer.scope.search_index_for_var(index,stdeallocate.parent.tag(),\
          name)
        if "pinned" in ivar["qualifiers"]:
            hipfort_deallocations.append(_render_deallocation_hipfort_f_str("hipHostFree",name,stat))
        elif "device" in ivar["qualifiers"]:
            hipfort_deallocations.append(_render_deallocation_hipfort_f_str("hipFree",name,stat))
        else:
            unchanged_deallocation_args.append(name)
    result = []
    if len(unchanged_deallocation_args):
        if stat != None:
            unchanged_deallocation_args.append("".join(["stat=",stat]))
        result.append("".join(["deallocate","(",",".join(unchanged_deallocation_args),")"]))
    result += hipfort_deallocations
    return (textwrap.indent("\n".join(result),indent), len(hipfort_deallocations))

def handle_use_statement_cuf(stuse, joined_statements, index):
    """Removes CUDA Fortran use statements and 
    adds hipfort use statements instead. 
    """
    mod, qualifiers, renamings, only = util.parsing.parse_use_statement(joined_statements)

    cuf_2_hipfort = {
      "cudafor" : "hipfort",
      "cublas" : "hipblas",
      "cusparse" : "hipsparse",
      "cufft" : "hipfft",
      "curand" : "hipblas",
    }

    # TODO handle only part
    cuf_module = mod in cuf_2_hipfort
    if cuf_module:
        assert isinstance(stuse.parent,nodes.STContainerBase)
        stuse.parent.add_use_statement(cuf_2_hipfort[mod])
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
        index = copy.deepcopy(indexer.scope.EMPTY)
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
        if stdeclaration.derived_type_parent == None:
            ivar = indexer.scope.search_index_for_var(
              index,stdeclaration.parent.tag(),
                var_name)
        else:
            parent_type = indexer.scope.search_index_for_type(
                index,stdeclaration.parent.tag(),
                stdeclaration.derived_type_parent)
            ivar = next(
                (v for v in parent_type["variables"] if v["name"] == var_name.lower()),
                None)
            if ivar == None:
                raise util.error.LookupError("no index record found for variable '{}' in scope".format(var_name))
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
                    raise util.error.LimitationError("device array without pointer or allocatable qualifier not supported in module")
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
