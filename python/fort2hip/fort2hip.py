# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import os
import copy
import logging

import addtoplevelpath
import translator.translator as translator
import indexer.indexer as indexer
import indexer.scoper as scoper
import scanner.scanner as scanner
import utils.logging
import utils.fileutils

fort2hip_dir = os.path.dirname(__file__)
exec(open(os.path.join(fort2hip_dir,"fort2hip_options.py.in")).read())
exec(open(os.path.join(fort2hip_dir,"fort2hip_filter.py.in")).read())
exec(open(os.path.join(fort2hip_dir,"fort2hip_codegen.py.in")).read())

#generate_launcher     = EMIT_KERNEL_LAUNCHER
#generate_cpu_launcher = generate_launcher and EMIT_CPU_IMPLEMENTATION
def render_loop_kernel(stkernel,
                       index,
                       current_cpp_file_ctx,
                       current_f03_file_ctx):
    utils.logging.log_enter_function(LOG_PREFIX,"render_loop_kernel")

    ttloopkernel = stkernel.parse_result
    scope        = scoper.create_scope(stkernel.parent_tag)

    utils.logging.log_debug3(LOG_PREFIX,"_intrnl_update_context_from_loop_kernels","parse result:\n```"+ttloopkernel.c_str().rstrip()+"\n```")

    # general
    kernel_name          = stkernel.kernel_name()
    kernel_launcher_name = stkernel.kernel_launcher_name()

    kernel_context = create_loop_kernel_context(stkernel.kernel_name(),
                                                ttloopkernel,
                                                scope,
                                                error_handling):
    # treat reduction_vars vars / acc default(present) vars


    hip_context["have_reductions"] = False # |= len(reduction_ops)

    fort2hip.model.render_hip_kernel_cpp(self.create_kernel_context("mykernel"))
    fort2hip.model.render_hip_kernel_launcher_cpp(self.create_kernel_context("mykernel",False),
                                                      self.create_hip_kernel_launcher_context("mykernel"))
    fort2hip.model.render_cpu_kernel_launcher_cpp(self.create_kernel_context("mykernel"),
                                                      self.create_cpu_kernel_launcher_context("mykernel"))

    fort2hip.model.render_kernel_launcher_f03(self.create_kernel_context("mykernel"),
                                              self.create_hip_kernel_launcher_context("mykernel"))

def render_device_procedure(stnode,index,current_cpp_file_ctx,current_f03_file_ctx):
    pass
def traverse_scanner_tree(stree,index,kernels_to_convert_to_hip=["*"]):
    cpp_file_contexts   = []
    f03_module_contexts = []
    current_cpp_file_ctx = None
    current_f03_file_ctx = None
    def traverse_node_(stnode)
        """Traverse the scanner tree and emit code generation context.
        """
        nonlocal index
        nonlocal kernels_to_convert_to_hip
        nonlocal cpp_file_contexts
        nonlocal f03_file_contexts
        nonlocal current_cpp_file_ctx
        nonlocal current_f03_file_ctx
        # main
        if isinstance(stnode, scanner.STRoot):
            pass
        elif loop_kernel_filter(stnode,kernels_to_convert_to_hip):
            render_loop_kernel(stnode,index,current_cpp_file_ctx,current_f03_file_ctx)
        elif device_procedure_filter(stnode,kernels_to_convert_to_hip): # handle before STProcedure (without attributes) is handled
            render_device_procedure(stnode,index,current_cpp_file_ctx,current_f03_file_ctx)
        elif isinstance(stnode,(scanner.STProgram,scanner.STModule,scanner.STProcedure)):
            stnode_name      = stnode.name.lower()
            stnode_is_module = isinstance(stnode,STModule)
            stnode_fort2hip_module_name = fort2hip_module_name(stnode)
            inode = next((irecord for irecord in index if irecord["name"] == stnode_name),None)
            if inode == None:
                utils.logging.log_error(LOG_PREFIX,\
                                        "traverse_scanner_tree",\
                                        "could not find index record for scanner tree node '{}'.".format(stnode_name))
                sys.exit() # TODO add error code

            itypes_local       = inode["types"] # local types
            local_used_modules = used_modules(stnode,inode)
            # derived_type_snippets = render_types(itypes)
            if stnode_is_module:
                module_used_modules = local_used_modules
            # no loop kernels in parent but device routines or global kernels
            current_cpp_file_ctx = copy.copy(EMPTY_CPP_FILE_CONTEXT)
            current_f03_file_ctx = copy.copy(EMPTY_F03_FILE_CONTEXT)
            for stchildnode in stnode.children:
               traverse_node_(stchildnode,index,kernels_to_convert_to_hip)
            cpp_file_contexts.append(current_cpp_file_ctx)
            f03_file_contexts.append(current_f03_file_ctx)
        return cpp_file_contexts, f03_file_contexts
