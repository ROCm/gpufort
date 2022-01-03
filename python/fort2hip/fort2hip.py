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
exec(open(os.path.join(fort2hip_dir,"fort2hip_render.py.in")).read())

def traverse_loop_kernels(stmodule,index,kernels_to_convert_to_hip=["*"]):
    def select_(kernel):
        nonlocal kernels_to_convert_to_hip
        if not len(kernels_to_convert_to_hip):
            return False
        else: 
            condition1 = not kernel.ignore_in_s2s_translation
            condition2 = \
                    kernels_to_convert_to_hip[0] == "*" or\
                    kernel.min_lineno() in kernels_to_convert_to_hip or\
                    kernel.kernel_name() in kernels_to_convert_to_hip
            return condition1 and condition2
    def loop_kernel_filter_(child):
        return isinstance(child, scanner.STLoopKernel) and select_(child)
    snippets = []
    for stloopkernel in stmodule.find_all(filter=loop_kernel_filter_, recursively=True):
        pass
    return snippets

def traverse_types(imodule,index):
    for stloopkernel in stmodule.find_all(filter=loop_kernel_filter_, recursively=True):
        pass
    return snippets

def traverse_program_unit(prefix,stprogramunit,iprogramunit):
    pass

def traverse_module(prefix):
    pass

def traverse_scanner_tree(stree,index):
    cpp_file_contexts   = []
    f03_module_contexts = []
    def traverse_node_(stnode)
        """Traverse the scanner tree and emit code generation context.
        """
        nonlocal index
        nonlocal cpp_file_contexts
        nonlocal f03_file_contexts
        # main 
        if isinstance(stnode, scanner.STRoot):
            for stchildnode in stnode.children:
                traverse_node_(stchildnode,index)
        elif isinstance(stnode,(scanner.STProgram,scanner.STModule,scanner.STProcedure)):
            stnode_name                 = stnode.name.lower()
            stnode_fort2hip_module_name = fort2hip_module_name(stnode)
            inode = next((irecord for irecord in index if irecord["name"] == stnode_name),None)
            if inode == None:
                utils.logging.log_error(LOG_PREFIX,\
                                        "traverse_scanner_tree",\
                                        "could not find index record for scanner tree node '{}'.".format(stnode_name))
                sys.exit() # TODO add error code

            itypes_local = inode["types"] # local types
            local_used_modules = used_modules(stnode,inode)
            # derived_type_snippets = render_types(itypes) 
            module_used_modules = local_used_modules if isinstance(stnode,STModule) else []
            # no loop kernels in parent but device routines or global kernels
            

            for stchildnode in stnode.children:
                traverse_node_(stchildnode)
            cpp_file_contexts.append({})


#def traverse_scanner_tree(stree,index,kernels_to_convert_to_hip=["*"])
#    """Traverse the scanner tree."""
#    def select_(kernel):
#        nonlocal kernels_to_convert_to_hip
#        if not len(kernels_to_convert_to_hip):
#            return False
#        else: 
#            condition1 = not kernel.ignore_in_s2s_translation
#            condition2 = \
#                    kernels_to_convert_to_hip[0] == "*" or\
#                    kernel.min_lineno() in kernels_to_convert_to_hip or\
#                    kernel.kernel_name() in kernels_to_convert_to_hip
#            return condition1 and condition2
#    def loop_kernel_filter_(child):
#        return isinstance(child, scanner.STLoopKernel) and select_(child)
#    #  TODO distinguish between device routines and global routines
#    def device_procedure_filter_(child):
#        return type(child) is scanner.STProcedure and\
#          child.must_be_available_on_device() and select_(child)

#def render_interopable_types(stcontainer,index):
#    pass
#
#def render_device_function()
#    pass
#
#def render_loop_kernel(stkernel,index):
#    pass
