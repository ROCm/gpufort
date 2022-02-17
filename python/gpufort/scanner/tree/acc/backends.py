# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import util
from .. import opts
from .. import nodes
from . import accnodes


class AccBackendBase:

    def __init__(self, stnode):
        self._stnode = stnode

    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[]):
        assert False, "not implemented"


class AccPostprocessBackendBase:

    def run(self, stree, index):
        """:param stree: the full scanner tree"""
        """:param staccdirectives: All acc directive tree accnodes."""
        #assert False, "not implemented"
        pass


ACC_BACKENDS = {}
ACC_LOOP_KERNEL_BACKENDS = {}
ACC_POSTPROCESS_BACKENDS = {}
ACC_ALLOCATE_BACKENDS = {}
ACC_DEALLOCATE_BACKENDS = {}


def register_acc_backend(name, directive_generator_class,
                         loop_kernel_generator_class, postprocess_class,
                         allocate_func, deallocate_func, runtime_module_name):
    if not name in nodes.SUPPORTED_DESTINATION_DIALECTS:
        nodes.SUPPORTED_DESTINATION_DIALECTS.append(name)
        nodes.RUNTIME_MODULE_NAMES[name] = runtime_module_name
        ACC_BACKENDS[name] = directive_generator_class
        ACC_LOOP_KERNEL_BACKENDS[name] = loop_kernel_generator_class
        ACC_POSTPROCESS_BACKENDS[name] = postprocess_class
        ACC_ALLOCATE_BACKENDS[name] = allocate_func
        ACC_DEALLOCATE_BACKENDS[name] = deallocate_func


def handle_allocate_acc(stallocate,
                        joined_statements,
                        index,
                        destination_dialect=""):
    indent = stallocate.first_line_indent()
    checked_dialect = nodes.check_destination_dialect(\
        opts.destination_dialect if not len(destination_dialect) else destination_dialect)
    if checked_dialect in ACC_ALLOCATE_BACKENDS:
        epilog = ACC_ALLOCATE_BACKENDS[checked_dialect](stallocate, index)
        for line in epilog:
            stallocate.add_to_epilog(line)
    return joined_statements, False


def handle_deallocate_acc(stdeallocate,
                          joined_statements,
                          index,
                          destination_dialect=""):
    indent = stdeallocate.first_line_indent()
    checked_dialect = nodes.check_destination_dialect(\
        opts.destination_dialect if not len(destination_dialect) else destination_dialect)
    if checked_dialect in ACC_DEALLOCATE_BACKENDS:
        prolog = ACC_DEALLOCATE_BACKENDS[checked_dialect](stdeallocate, index)
        for line in prolog:
            stdeallocate.add_to_prolog(line)
    return joined_statements, False


@util.logging.log_entry_and_exit(opts.log_prefix)
def postprocess_tree_acc(stree, index, destination_dialect=""):
    """Add use statements."""
    checked_dialect = nodes.check_destination_dialect(\
        opts.destination_dialect if not len(destination_dialect) else destination_dialect)
    if checked_dialect in ACC_POSTPROCESS_BACKENDS:

        def directive_filter(node):
            return isinstance(node,accnodes.STAccDirective) and\
                   not node.ignore_in_s2s_translation

        directives = stree.find_all(filter=directive_filter, recursively=True)
        for directive in directives:
            stnode = directive.parent.first_entry_in_decl_list()
            # add acc use statements
            if not stnode is None:
                indent = stnode.first_line_indent()
                acc_runtime_module_name = nodes.RUNTIME_MODULE_NAMES[
                    checked_dialect]
                if acc_runtime_module_name != None and len(
                        acc_runtime_module_name):
                    stnode.add_to_prolog(
                        "{0}use {1}\n{0}use iso_c_binding\n".format(
                            indent, acc_runtime_module_name))
        #if type(directive.parent
        # call backend
        ACC_POSTPROCESS_BACKENDS[checked_dialect]().run(stree, index)
