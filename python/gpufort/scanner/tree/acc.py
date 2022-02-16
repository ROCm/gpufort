# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from .. import opts
from . import base
from gpufort import util

class AccBackendBase:
    def __init__(self,stnode):
        self._stnode = stnode
    def transform(self,joined_lines,joined_statements,statements_fully_cover_lines,index=[]):
        assert False, "not implemented"

class AccPostprocessBackendBase:
    def run(self,stree,index):
        """:param stree: the full scanner tree"""
        """:param staccdirectives: All acc directive tree nodes."""
        #assert False, "not implemented"
        pass

ACC_BACKENDS             = {} 
ACC_LOOP_KERNEL_BACKENDS = {} 
ACC_POSTPROCESS_BACKENDS = {}
ACC_ALLOCATE_BACKENDS    = {}
ACC_DEALLOCATE_BACKENDS  = {}

def register_acc_backend(name,
                         directive_generator_class,
                         loop_kernel_generator_class,
                         postprocess_class,
                         allocate_func,
                         deallocate_func,
                         runtime_module_name):
    if not name in base.SUPPORTED_DESTINATION_DIALECTS:
        base.SUPPORTED_DESTINATION_DIALECTS.append(name)
        base.RUNTIME_MODULE_NAMES[name]     = runtime_module_name
        ACC_BACKENDS[name]             = directive_generator_class 
        ACC_LOOP_KERNEL_BACKENDS[name] = loop_kernel_generator_class
        ACC_POSTPROCESS_BACKENDS[name] = postprocess_class
        ACC_ALLOCATE_BACKENDS[name]    = allocate_func
        ACC_DEALLOCATE_BACKENDS[name]  = deallocate_func

class STAccDirective(base.STDirective):
    """Class for handling ACC directives."""
    def __init__(self,first_linemap,first_linemap_first_statement,directive_no):
        base.STDirective.__init__(self,first_linemap,first_linemap_first_statement,directive_no,sentinel="!$acc")
        self._default_present_vars = []
    def find_substring(self,token):
        return token in self.first_statement()
    def find_any_substring(self,tokens):
        result = False
        for token in tokens:
            result = result or self.find_substring(token)
        return result
    def find_all_substrings(self,tokens):
        result = True
        for token in tokens:
            result = result and self.find_substring(token)
        return result
    def is_end_directive(self):
        return self.find_substring("acc end")
    def is_data_directive(self):
        return self.find_substring("acc data")
    def is_enter_data_directive(self):
        return self.find_substring("acc enter")
    def is_exit_data_directive(self):
        return self.find_substring("acc exit")
    def is_init_directive(self):
        return self.find_substring("acc init")
    def is_shutdown_directive(self):
        return self.find_substring("acc shutdown")
    def is_update_directive(self):
        return self.find_substring("acc update")
    def is_wait_directive(self):
        return self.find_substring("acc wait")
    def is_loop_directive(self):
        return self.find_substring("acc loop")
    def is_kernels_directive(self):
        return not self.find_substring("acc kernels loop") and\
            self.find_substring("acc kernels")
    def is_parallel_directive(self):
        return not self.find_substring("acc parallel loop") and\
            self.find_substring("acc parallel")
    def is_loop_directive(self):
        return self.find_substring("acc loop")
    def is_parallel_loop_directive(self):
        return self.find_substring("acc parallel loop")
    def is_kernels_loop_directive(self):
        return self.find_substring("acc kernels loop")
    def is_declare_directive(self):
        return self.find_substring("acc declare")
    def is_purely_declarative(self):
        return self.is_declare_directive() or\
               self.find_substring("acc routine")
    def __str__(self):
        return """
{{ single_line_statement={single_line_statement},
         is_init_directive={is_init_directive},
         is_shutdown_directive={is_shutdown_directive},
         is_end_directive={is_end_directive},
         is_enter_data_directive={is_enter_data_directive},
         is_exit_data_directive={is_exit_data_directive},
         is_wait_directive={is_wait_directive},
         is_loop_directive={is_loop_directive},
         is_parallel_directive={is_parallel_directive},
         is_kernels_directive={is_kernels_directive},
         is_parallel_loop_directive={is_parallel_loop_directive} }}
""".format(
         single_line_statement=self.first_statement(),
         is_init_directive=self.is_init_directive(),
         is_shutdown_directive=self.is_shutdown_directive(),
         is_end_directive=self.is_end_directive(),
         is_enter_data_directive=self.is_enter_data_directive(),
         is_exit_data_directive=self.is_exit_data_directive(),
         is_wait_directive=self.is_wait_directive(),
         is_loop_directive=self.is_loop_directive(),
         is_parallel_directive=self.is_parallel_directive(),
         is_kernels_directive=self.is_kernels_directive(),
         is_parallel_loop_directive=self.is_parallel_loop_directive()
         ).strip().replace("\n","")
    __repr__ = __str__ 
    def transform(self,joined_lines,joined_statements,statements_fully_cover_lines,index=[]):
        if self.is_purely_declarative():
            return base.STNode.transform(self,joined_lines,joined_statements,statements_fully_cover_lines,index) 
        else:
            checked_dialect = base.check_destination_dialect(opts.destination_dialect)
            return ACC_BACKENDS[checked_dialect](self).transform(\
                    joined_lines,joined_statements,statements_fully_cover_lines,index)
class STAccLoopNest(STAccDirective,base.STLoopNest):
    def __init__(self,first_linemap,first_linemap_first_statement,directive_no):
        STAccDirective.__init__(self,first_linemap,first_linemap_first_statement,directive_no)
        base.STLoopNest.__init__(self,first_linemap,first_linemap_first_statement)
    def transform(self,joined_lines,joined_statements,statements_fully_cover_lines,index=[],destination_dialect=""):
        """
        :param destination_dialect: allows to override default if this kernel
                                   should be translated via another backend.
        """
        checked_dialect = base.check_destination_dialect(\
            opts.destination_dialect if not len(destination_dialect) else destination_dialect)
        return ACC_LOOP_KERNEL_BACKENDS[checked_dialect](self).transform(\
                joined_lines,joined_statements,statements_fully_cover_lines,index)

def handle_allocate_acc(stallocate,joined_statements,index,destination_dialect=""):
    indent = stallocate.first_line_indent() 
    checked_dialect = base.check_destination_dialect(\
        opts.destination_dialect if not len(destination_dialect) else destination_dialect)
    if checked_dialect in ACC_ALLOCATE_BACKENDS:
        epilog = ACC_ALLOCATE_BACKENDS[checked_dialect](stallocate,index)
        for line in epilog:
            stallocate.add_to_epilog(line)
    return joined_statements, False

def handle_deallocate_acc(stdeallocate,joined_statements,index,destination_dialect=""):
    indent = stdeallocate.first_line_indent() 
    checked_dialect = base.check_destination_dialect(\
        opts.destination_dialect if not len(destination_dialect) else destination_dialect)
    if checked_dialect in ACC_DEALLOCATE_BACKENDS:
        prolog = ACC_DEALLOCATE_BACKENDS[checked_dialect](stdeallocate,index)
        for line in prolog:
            stdeallocate.add_to_prolog(line)
    return joined_statements, False

@util.logging.log_entry_and_exit(opts.log_prefix)
def postprocess_tree_acc(stree,index,destination_dialect=""):
    """Add use statements."""
    checked_dialect = base.check_destination_dialect(\
        opts.destination_dialect if not len(destination_dialect) else destination_dialect)
    if checked_dialect in ACC_POSTPROCESS_BACKENDS: 
        def directive_filter(node):
            return isinstance(node,STAccDirective) and\
                   not node.ignore_in_s2s_translation
        directives = stree.find_all(filter=directive_filter, recursively=True)
        for directive in directives:
             stnode = directive.parent.first_entry_in_decl_list()
             # add acc use statements
             if not stnode is None:
                 indent = stnode.first_line_indent()
                 acc_runtime_module_name = base.RUNTIME_MODULE_NAMES[checked_dialect]
                 if acc_runtime_module_name != None and len(acc_runtime_module_name):
                     stnode.add_to_prolog("{0}use {1}\n{0}use iso_c_binding\n".format(indent,acc_runtime_module_name))
            #if type(directive.parent
        # call backend
        ACC_POSTPROCESS_BACKENDS[checked_dialect]().run(stree,index)