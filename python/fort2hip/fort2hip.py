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

def lookup_index_entries_for_vars_in_kernel_body(scope,
                                                 all_vars,
                                                 reductions,
                                                 shared_vars,
                                                 local_vars,
                                                 error_handling=None):
    Lookup index variables
    """
    :param list all_vars: List of all variable expressions (var)
    :param list reductions: List of tuples pairing a reduction operation with the associated
                             variable expressions
    :param list shared:     List of variable expressions that are shared by the workitems/threads in a workgroup/threadblock
    :param list local_vars: List of variable expressions that can be mapped to local variables per workitem/thread
    :param str error_handling: Emit error messages if set to 'strict'; else emit warning
    :note: Emits errors (or warning) if a variable in another list is not present in all_vars
    :note: Emits errors (or warning) if a reduction variable is part of a struct.
    :return: A tuple containing (in this order): global variables, reduced global variables, shared variables, local variables
             as list of index entries. The reduced global variables have an extra field 'op' that
             contains the reduction operation.
    """
    utils.logging.log_enter_function(LOG_PREFIX,"lookup_index_entries_for_variables")

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
    rglobal_reduced_vars = []
    iglobal_vars         = []

    for reduction_op,var_exprs in reductions.items():
        for var_expr in var_exprs:
            if "%" in var_expr:
               if error_handling=="strict":
                   pass # TODO error
               else:
                   pass # TOOD warn
            else:
               ivar, _ = scoper.search_scope_for_variable(scope,var_expr,error_handling)
               rvar    = copy.deepcopy(ivar)
               rvar["op"] = reduction_op
               global_reduced_vars.append(rvar)
            try:
                all_vars2.remove(var_expr)
            except:
                pass # TODO error
    for var_expr in all_vars2:
        ivar, _ = scoper.search_scope_for_variable(scope,var_expr,error_handling)
        global_vars.append(ivar)

    utils.logging.log_leave_function(LOG_PREFIX,"lookup_index_entries_for_variables")
    return iglobal_vars, rglobal_reduced_vars, ishared_vars, ilocal_vars

def create_kernel_base_context(kernel_name,
                               c_body)
    global GET_LAUNCH_BOUNDS
    launch_bounds = GET_LAUNCH_BOUNDS(kernel_name)
    return {
           "name"          : kernel_name,
           "launch_bounds" : "" if launch_bounds == None else launch_bounds,
           "c_body"        : c_body,
           }

#ttloopkernel = stloopkernel.parse_result
#scope        = scoper.create_scope(index,
#                                   stloopkernel.parent.tag())
def create_loop_kernel_context(kernel_name,
                               ttloopkernel,
                               scope,
                               error_handling=None):
    kernel = create_kernel_base_context(kernel_name,
                                        ttloopkernel.c_str())
    kernel["global_vars"],
    kernel["global_reduced_vars"],
    kernel["shared_vars"],
    kernel["local_vars"] = lookup_index_entries_for_variables(scope,
                                                              ttloopkernel.vars_in_body(),
                                                              ttloopkernel.gang_team_reductions(),
                                                              ttloopkernel.gang_team_shared_vars(),
                                                              ttloopkernel.local_scalars(),
                                                              index,
                                                              error_handling)
    return kernel

def create_kernel_launcher_base_context(kernel_name,
                                        kind,
                                        problem_size,
                                        debug_output):
    return {
           "kind"         : kind,
           "name"         : "_".join(["launch",kernel_name,kind]),
           "used_modules" : [],
           "problem_size" : problem_size,
           "debug_output" : debug_output,
           }

def create_hip_kernel_launcher_context(kernel_name,
                                       kind,
                                       grid,
                                       block,
                                       problem_size,
                                       debug_output):
    kernel_launcher = create_kernel_launcher_base_context(kernel_name,
                                                          kind,
                                                          problem_size,
                                                          debug_output)
    if kind == "hip":
        kernel_launcher["grid"]  = grid
        kernel_launcher["block"] = block
    return kernel_launcher
def create_cpu_kernel_launcher_context(self,
                                       kernel_name):
    # for launcher interface
    kernel_launcher = {}
    kernel_launcher["name"] = "_".join(["launch",kernel_name,"cpu"])
    kernel_launcher["debug_output"]  = True
    kernel_launcher["kind"]          = "cpu"
    kernel_launcher["used_modules"]  = []
    return kernel_launcher

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
