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

class HipCodeGenerator(CodeGenerator):
    def _fort2x_module_name(self,
                            stnode):
        """Name of module file generated by fort2x for module/program/procedure.
        """
        return "{}{}".format(stnode.tag().replace(":","_"))

class CodeGenerator:
    def _fort2x_module_name(self,
                            stnode):
        """Name of module file generated by fort2x for module/program/procedure.
        """
        assert False, "To be implemented by subclass"
    def _parent_fort2x_modules(self,
                               stnode):
        if not stnode.parent == None and not isinstance(stnode.parent,scanner.STRoot):
            return [{ "name": fort2x_module_name(stnode.parent.name.lower), "only": [] }]
        else:
            return []
    
    def _used_modules(self,
                      stnode,
                      inode):
       used_modules = [{"name": mod,"only:"[]} for mod in ["gpufort_array"]
       used_modules += inode["used_modules"] # local modules
       used_modules += parent_fort2x_modules(stnode)  
       return used_modules

    def _render_loop_kernel(self,
                            stkernel,
                            index,
                            cppfilegen,
                            f03filegen):
        utils.logging.log_enter_function(LOG_PREFIX,"self.__render_loop_kernel")
    
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
    
    
        hip_context["have_reductions"] = False # |= len(reduction_ops)
    
    def render_device_procedure(stnode,
                                index,
                                current_cppfilegen,
                                current_f03filegen):
        pass
    def traverse_scanner_tree(stree,
                              index,
                              kernels_to_convert_to_hip=["*"]):
        cpp_file_contexts   = []
        f03_module_contexts = []
        current_cppfilegen = None
        current_f03filegen = None
        def traverse_node_(stnode)
            """Traverse the scanner tree and emit code generation context.
            """
            nonlocal index
            nonlocal kernels_to_convert_to_hip
            nonlocal cpp_file_contexts
            nonlocal f03_file_contexts
            nonlocal current_cppfilegen
            nonlocal current_f03filegen
            # main
            if isinstance(stnode,scanner.STRoot):
                pass
            elif loop_kernel_filter(stnode,
                                    kernels_to_convert_to_hip):
                self._render_loop_kernel(stnode,
                                         index,
                                         current_cppfilegen,
                                         current_f03filegen)
            elif device_procedure_filter(stnode,
                                         kernels_to_convert_to_hip): # handle before STProcedure (without attributes) is handled
                inode = next((irecord for irecord in index if irecord["name"] == stnode_name),None)
                self._render_device_procedure(stnode,
                                              inode,
                                              index,
                                              current_cppfilegen,
                                              current_f03filegen)
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
                current_cppfilegen = copy.copy(EMPTY_CPP_FILE_CONTEXT)
                current_f03filegen = copy.copy(EMPTY_F03_FILE_CONTEXT)
                for stchildnode in stnode.children:
                   traverse_node_(stchildnode,
                                  index,
                                  kernels_to_convert_to_hip)
                cpp_file_contexts.append(current_cppfilegen)
                f03_file_contexts.append(current_f03filegen)
            return cpp_file_contexts, f03_file_contexts