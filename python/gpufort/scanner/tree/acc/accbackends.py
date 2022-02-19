# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import util
from ... import opts
from .. import backends
from .. import nodes
from . import accnodes

DIRECTIVE_BACKENDS = {}
LOOP_KERNEL_BACKENDS = {}
POSTPROCESS_BACKENDS = {}
ALLOCATE_BACKENDS = {}
DEALLOCATE_BACKENDS = {}


def register_acc_backend(name, directive_generator_class,
                         loop_kernel_generator_class, postprocess_class,
                         allocate_func, deallocate_func):
    if not name in backends.SUPPORTED_DESTINATION_DIALECTS:
        backends.SUPPORTED_DESTINATION_DIALECTS.append(name)
        DIRECTIVE_BACKENDS[name] = directive_generator_class
        LOOP_KERNEL_BACKENDS[name] = loop_kernel_generator_class
        POSTPROCESS_BACKENDS[name] = postprocess_class
        ALLOCATE_BACKENDS[name] = allocate_func
        DEALLOCATE_BACKENDS[name] = deallocate_func


class AccBackendBase:

    def __init__(self, stnode):
        self._stnode = stnode

    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[]):
        assert False, "not implemented"


@util.logging.log_entry_and_exit(opts.log_prefix)
def handle_allocate_acc(stallocate, joined_statements, index, dest_dialect=""):
    epilog = ALLOCATE_BACKENDS[dest_dialect](stallocate, index)
    for line in epilog:
        stallocate.add_to_epilog(line)
    return joined_statements, False


@util.logging.log_entry_and_exit(opts.log_prefix)
def handle_deallocate_acc(stdeallocate,
                          joined_statements,
                          index,
                          dest_dialect=""):
    prolog = DEALLOCATE_BACKENDS[dest_dialect](stdeallocate, index)
    for line in prolog:
        stdeallocate.add_to_prolog(line)
    return joined_statements, False


@util.logging.log_entry_and_exit(opts.log_prefix)
def add_runtime_module_use_statements(acc_runtime_module_name):

    def directive_filter(node):
        return isinstance(node,accnodes.STAccDirective) and\
               not node.ignore_in_s2s_translation

    directives = stree.find_all(filter=directive_filter, recursively=True)
    for directive in directives:
        stnode = directive.parent.first_entry_in_decl_list()
        # add acc use statements
        if not stnode is None:
            indent = stnode.first_line_indent()
            if acc_runtime_module_name != None and len(
                    acc_runtime_module_name):
                stnode.add_to_prolog(
                    "{0}use {1}\n{0}use iso_c_binding\n".format(
                        indent, acc_runtime_module_name))


@util.logging.log_entry_and_exit(opts.log_prefix)
def postprocess_tree_acc(stree, index, dest_dialect):
    """Add use statements."""
    # call backend
    if dest_dialect in POSTPROCESS_BACKENDS.keys():
        POSTPROCESS_BACKENDS[dest_dialect](stree, index)


nodes.STAllocate.register_backend("acc", "hip", handle_allocate_acc)
nodes.STDeallocate.register_backend("acc", "hip", handle_deallocate_acc)
backends.register_postprocess_backend("acc", "hip", postprocess_tree_acc)
