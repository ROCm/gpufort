# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import util
from ... import opts
from .. import nodes
from .. import grammar
from .. import backends
from . import cufnodes

LOOP_KERNEL_BACKENDS = {}


class CufBackendBase:

    def __init__(self, stnode):
        self._stnode = stnode


def register_cuf_backend(name, loop_kernel_generator_class):
    if not name in backends.SUPPORTED_DESTINATION_DIALECTS:
        backends.SUPPORTED_DESTINATION_DIALECTS.append(name)
        LOOP_KERNEL_BACKENDS[name] = loop_kernel_generator_class


def handle_allocate_cuf(stallocate, joined_statements, index):
    indent = stallocate.first_line_indent()
    # CUF
    transformed = False
    bytes_per_element = []
    array_qualifiers = []
    for array_name in stallocate.parse_result.variable_names():
        ivar,_  = scope.search_index_for_variable(index,stallocate.parent.tag(),\
          array_name)
        bytes_per_element.append(ivar["bytes_per_element"])
        qualifier, transformation_required = nodes.pinned_or_on_device(ivar)
        transformed |= transformation_required
        array_qualifiers.append(qualifier)
    subst = stallocate.parse_result.hip_f_str(bytes_per_element,
                                              array_qualifiers,
                                              indent=indent).lstrip(" ")
    return (subst, transformed)


def handle_deallocate_cuf(stdeallocate, joined_statements, index):
    indent = stdeallocate.first_line_indent()
    transformed = False
    array_qualifiers = []
    for array_name in stdeallocate.parse_result.variable_names():
        ivar,_  = scope.search_index_for_variable(index,stdeallocate.parent.tag(),\
          array_name)
        on_device = nodes.index_variable_is_on_device(ivar)
        qualifier, transformed1 = nodes.pinned_or_on_device(ivar)
        transformed |= transformed1
        array_qualifiers.append(qualifier)
    subst = stdeallocate.parse_result.hip_f_str(array_qualifiers,
                                                indent=indent).lstrip(" ")
    return (subst, transformed)


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
            indent = self.first_line_indent()
            last_decl_list_node.add_to_epilog(
                "{0}type(c_ptr) :: hipblasHandle = c_null_ptr\n".format(
                    indent))

            local_cublas_calls = call.parent.find_all(filter=has_cublas_call,
                                                      recursively=False)
            first = local_cublas_calls[0]
            indent = self.first_line_indent()
            first.add_to_prolog(
                "{0}hipblasCreate(hipblasHandle)\n".format(indent))
            last = local_cublas_calls[-1]
            indent = self.first_line_indent()
            last.add_to_epilog(
                "{0}hipblasDestroy(hipblasHandle)\n".format(indent))


@util.logging.log_entry_and_exit(opts.log_prefix)
def postprocess_tree_cuf(stree, index, dest_dialect):
    _handle_cublas_v1(stree)


nodes.STAllocate.register_backend("cuf", "hip", handle_allocate_cuf)
nodes.STDeallocate.register_backend("cuf", "hip", handle_deallocate_cuf)
backends.register_postprocess_backend("cuf", "hip", postprocess_tree_cuf)
