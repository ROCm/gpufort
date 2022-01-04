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

# scope          = create_scope(index,stloopkernel.parent.tag())
# ttloopkernel   = stloopkernel.parse_result
# all_vars       = ttloopkernel.vars_in_body()
# reduction_vars = ttloopkernel.gang_team_reductions()
# shared_vars    = ttloopkernel.gang_team_shared_vars()
# local_vars     = ttloopkernel.local_scalars()
def lookup_index_entries_for_variables(scope,all_vars,reduction_vars,shared_vars,local_vars,index,error_handling=None):
    # TODO check which variable expression points to a struct member, identify top-level struct and pass only that one
    # can be done as preprocessing step
    # possible issue with struct members that are reduced as we lose information in this step
    # pass reduced variables as extra argument put original expression into rvar?
    def strip_member_access_(var_exprs):
        result = []
        return [var_expr.split("%")[0] for var_expr in var_exprs]
    all_vars2 = strip_member_access_(all_vars)

    def lookup_index_vars_(var_exprs):
        nonlocal all_vars2
        ivars = []
        for var_expr in var_exprs:
            ivar, _ = scoper.search_scope_for_variable(scope,var_expr,error_handling)
            ivars.append(ivar) 
            try:
                all_vars2.remove(var_expr)
            except:
                pass # TODO error
        return ivars

    ilocal_vars          = lookup_index_vars_(strip_member_access_(local_vars))
    ishared_vars         = lookup_index_vars_(strip_member_access_(shared_vars))
    iglobal_reduced_vars = []
    iglobal_vars         = []
    
    for reduction_op,var_exprs in reduction_vars.items():
        ivar, _ = scoper.search_scope_for_variable(scope,var_expr,error_handling)
        rvars = copy.deepcopy(ivar)
        rvar["op"] = reduction_op
        global_reduced_vars.append(rvar)
        try:
            all_vars.remove(var_expr)
        except:
            pass # TODO error
    for var_expr in all_vars:
        ivar, _ = scoper.search_scope_for_variable(scope,var_expr,error_handling)
        global_vars.append(ivar)
    return global_vars, global_reduced_vars, shared_vars, local_vars
 
def create_loop_kernel_context(stloopkernel,kernel_name):
    local_vars          = stloopkernel.local_scalars()
    shared_vars         = [] 
    global_reduced_vars = stloopkernel.gang_team_reductions()
    global_vars         =  
    
    kernel = {}
    kernel["launch_bounds"]       = "__launch_bounds___(1024,1)"
    kernel["name"]                = kernel_name
    kernel["shared_vars"]         = shared_vars
    kernel["local_vars"]          = local_vars
    kernel["global_vars"]         = global_vars
    kernel["global_reduced_vars"] = global_reduced_vars
    kernel["c_body"]              = "// do nothing" 
    kernel["f_body"]              = "! do nothing"
    return kernel

def render_loop_kernel(stkernel,index,current_cpp_file_ctx,current_f03_file_ctx):
    utils.logging.log_enter_function(LOG_PREFIX,"render_loop_kernel")
    
    generate_launcher     = EMIT_KERNEL_LAUNCHER
    generate_cpu_launcher = generate_launcher and EMIT_CPU_IMPLEMENTATION
    
    parse_result = stkernel.parse_result
    parent_tag   = stkernel.parent.tag()
    scope        = scoper.create_scope(index,parent_tag)

    #    kernel_args, c_kernel_local_vars, macros, input_arrays, local_cpu_routine_args =\
    #      _intrnl_derive_kernel_arguments(scope,\
    #        parse_result.vars_in_body(),\
    #        parse_result.local_scalars(),\
    #        parse_result.loop_vars(),\
    #        True, parse_result.deviceptrs())
        
    utils.logging.log_debug3(LOG_PREFIX,"_intrnl_update_context_from_loop_kernels","parse result:\n```"+parse_result.c_str().rstrip()+"\n```")

    # general
    kernel_name          = stkernel.kernel_name()
    kernel_launcher_name = stkernel.kernel_launcher_name()
   
    # treat reduction_vars vars / acc default(present) vars
    hip_context["have_reductions"] = False # |= len(reduction_ops)
    kernel_call_arg_names     = []
    cpu_kernel_call_arg_names = []
    reductions                = parse_result.gang_team_reductions(translator.make_c_str)
    reduction_vars            = []
    
    fort2hip.model.render_hip_kernel_hip_cpp(self.create_kernel_context("mykernel"))
    fort2hip.model.render_hip_kernel_launcher_hip_cpp(self.create_kernel_context("mykernel",False),
                                                      self.create_hip_kernel_launcher_context("mykernel"))
    fort2hip.model.render_cpu_kernel_launcher_hip_cpp(self.create_kernel_context("mykernel"),
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
