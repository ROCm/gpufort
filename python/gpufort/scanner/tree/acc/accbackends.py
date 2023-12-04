# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import util

from ... import opts

from . import accnodes

class AccBackendBase:

    def __init__(self):
        self.stnode = None

    def configure(self,stnode):
        self.stnode = stnode

    def transform(self,*args,**kwargs):
        assert False, "not implemented"

@util.logging.log_entry_and_exit(opts.log_prefix)
def add_runtime_module_use_statements(stree,acc_runtime_module_name):

    def directive_filter(node):
        return isinstance(node,accnodes.STAccDirective) and\
               not node.ignore_in_s2s_translation

    directives = stree.find_all(filter=directive_filter, recursively=True)
    for directive in directives:
        directive.parent.add_use_statement(acc_runtime_module_name)
        directive.parent.add_use_statement("iso_c_binding")
