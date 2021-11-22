# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import os
import copy
import logging

import addtoplevelpath
import fort2hip.model as model
import translator.translator as translator
import indexer.indexer as indexer
import indexer.scoper as scoper
import scanner.scanner as scanner
import utils.logging
import utils.fileutils

INDEXER_ERROR_CODE = 1000

def GET_DEFAULT_BLOCK_DIMS(kernel_name,dim):
    block_dims = { 1 : [128], 2 : [128,1,1], 3: [128,1,1] }
    return block_dims[dim]

def GET_DEFAULT_LAUNCH_BOUNDS(kernel_name):
    return None

fort2hip_dir = os.path.dirname(__file__)
exec(open("{0}/fort2hip_options.py.in".format(fort2hip_dir)).read())

def _intrnl_convert_dim3(dim3,dimensions,do_filter=True):
     result = []
     specified = dim3
     if do_filter:
         specified = [ x for x in dim3 if type(x) != int or x > 0 ]
     for i,value in enumerate(specified):
          if i >= dimensions:
              break
          el = {}
          el["dim"]   = chr(ord("X")+i)
          el["value"] = value
          result.append(el)
     return result

EMPTY_ARG = {
  "name"              : "", 
  "callarg_name"      : "", 
  "qualifiers"        : "", 
  "type"              : "", 
  "orig_type"         : "", 
  "c_type"            : "", 
  "c_size"            : "", 
  "c_value"           : "", 
  "c_suffix"          : "", 
  "is_array"          : "", 
  "reduction_op"      : "", 
  "bytes_per_element" : "", 
}

# arg for kernel generator
# array is split into multiple args
def _intrnl_init_arg(argname,f_type,kind,qualifiers=[],c_type="",is_array=False):
    f_type_final = f_type
    if len(kind):
        f_type_final += "({})".format(kind)
    arg = dict(EMPTY_ARG)
    arg["name"]         = argname
    arg["callarg_name"] = argname
    arg["qualifiers"]   = qualifiers
    arg["type"]         = f_type_final
    arg["orig_type"]    = f_type_final
    arg["c_type"]       = c_type
    arg["is_array"]     = is_array
    arg["bytes_per_element"] = translator.num_bytes(f_type,kind,default="-1")
    if not len(c_type):
        arg["c_type"] = translator.convert_to_c_type(f_type,kind,"void")
    if is_array:
        arg["c_type"] += " * __restrict__"
    return arg

def _intrnl_search_derived_types(imodule,search_subprograms=True):
    # 1. find all types and tag them
    global LOG_PREFIX

    utils.logging.log_enter_function(LOG_PREFIX,"_intrnl_search_derived_types",\
      {"modulename":imodule.get("name",""),\
       "search_subprograms":str(search_subprograms)})
    
    itypes = {}
    prefix = ""   
    def collect_types_(irecord):
        nonlocal itypes
        nonlocal prefix
        prefix += irecord["name"]
        for itype in irecord["types"]:
            ident = prefix + "_" + itype["name"]
            itypes[ident] = itype
        if search_subprograms:
            for isubprogram in irecord["subprograms"]:
                collect_types_(isubprogram)
    collect_types_(imodule)

    utils.logging.log_leave_function(LOG_PREFIX,"_intrnl_search_derived_types",
      {"typenames":str([itype["name"] for itype in itypes.values()])})
    
    return itypes

def _intrnl_update_context_from_derived_types(itypes,hip_context,f_context):
    global LOG_PREFIX

    utils.logging.log_enter_function(LOG_PREFIX,"_intrnl_update_context_from_derived_types",
      {"typenames":str([itype["name"] for itype in itypes.values()])})
    
    for ident,itype in itypes.items():
        interop_type = {}
        interop_type["f_name"]  = itype["name"]+"_interop"
        interop_type["c_name"]  = itype["name"]
        interop_type["members"] = []
        for ivar in itype["variables"]:
            f_type_final = ivar["f_type"]
            kind         = ivar["kind"]
            if len(kind):
                f_type_final += "({})".format(kind)
            #
            member = dict(EMPTY_ARG)
            member_name      = ivar["name"]
            member["name"]   = member_name
            member["c_type"] = ivar["c_type"]
            member["type"]   = f_type_final
            if ivar["rank"] > 0 and not ivar["f_type"]=="type":
                member["type"]   = "type(c_ptr)"
                member["c_type"] += "*"
            elif ivar["f_type"]=="type":
                member["type"]   = "type(c_ptr)"
                member["c_type"] = kind + "*"
                
            interop_type["members"].append(member)
            bound_members, count_members = [], []
            for d in range(1,ivar["rank"]+1):
                 # lower bounds
                 bound_member = _intrnl_init_arg("{}_lb{}".format(member_name,d),"integer","c_int",[],"int")
                 bound_member["callmember_name"] = "lbound({},{})".format(member_name,d)
                 bound_members.append(bound_member)
                 # number of elements per dimensions
                 count_member = _intrnl_init_arg("{}_n{}".format(member_name,d),"integer","c_int",[],"int")
                 count_member["callmember_name"] = "size({},{})".format(member_name,d)
                 count_members.append(count_member)
            interop_type["members"] += count_members + bound_members
        f_context["types"].append(interop_type)
        hip_context["types"].append(interop_type)
    
    utils.logging.log_leave_function(LOG_PREFIX,"_intrnl_update_context_from_derived_types")

def _intrnl_create_argument_context(ivar,argname,deviceptr_names=[],is_loop_kernel_arg=False):
    """
    Create an argument context dictionary based on a indexed variable.

    :param ivar: A variable description provided by the indexer.
    :type ivar: STDeclaration
    :return: a dicts containing Fortran `type` and `qualifiers` (`type`, `qualifiers`), C type (`c_type`), and `name` of the argument
    :rtype: dict
    """
    arg = _intrnl_init_arg(argname,ivar["f_type"],ivar["kind"],[ "value" ],"",ivar["rank"]>0)
    arg["bytes_per_element"] = ivar["bytes_per_element"] # scope value might be more accurate
    # TODO more kind and bytes per element should be obtained from scope var ivar as it can resolve them up to selected_kind parameters
    if "parameter" in ivar["qualifiers"] and not ivar["value"] is None:
        arg["c_value"] = ivar["value"] 
    lbound_args = []  # additional arguments that we introduce if variable is an array
    count_args      = []
    macro          = None
    # treat arrays
    rank = ivar["rank"] 
    if rank > 0:
        if argname in deviceptr_names:
            arg["callarg_name"] = "c_loc({})".format(argname)
        else: 
            arg["callarg_name"] = scanner.dev_var_name(argname)
        arg["type"]       = "type(c_ptr)"
        arg["qualifiers"] = [ "value" ]
        for d in range(1,rank+1):
             # lower bounds
             bound_arg = _intrnl_init_arg("{}_lb{}".format(argname,d),"integer","c_int",["value","intent(in)"],"const int")
             bound_arg["callarg_name"] = "lbound({},{})".format(argname,d)
             lbound_args.append(bound_arg)
             # number of elements per dimensions
             count_arg = _intrnl_init_arg("{}_n{}".format(argname,d),"integer","c_int",["value","intent(in)"],"const int")
             count_arg["callarg_name"] = "size({},{})".format(argname,d)
             count_args.append(count_arg)
        # create macro expression
        if is_loop_kernel_arg and not ivar["unspecified_bounds"]:
            macro = { "expr" : ivar["index_macro"] }
        else:
            macro = { "expr" : ivar["index_macro_with_placeholders"] }
    return arg, lbound_args, count_args, macro

def _intrnl_derive_kernel_arguments(scope, varnames, local_vars, loop_vars, is_loop_kernel_arg=False, deviceptr_names=[]):
    """
    Derive code generation contexts for the different interfaces and subroutines that 
    are generated by the fort2hip module.

    :param: varnames a list of Fortran varnames or derived type members such as 'a%b%c'
    """
    utils.logging.log_enter_function(LOG_PREFIX,"_intrnl_derive_kernel_arguments",{"varnames":str(varnames)})
    
    kernel_args, unknown_args, c_kernel_local_vars, macros = [], [], [], []
    local_args, local_cpu_routine_args, input_arrays       = [], [], []
    
    def include_arg_(name):
        name_lower = name.lower().strip()
        # Fortran var names never start with _; can be exploited when modifying code
        if name_lower.startswith("_") or\
           name_lower == "dim3" or\
           name_lower in translator.DEVICE_PREDEFINED_VARIABLES:
            return False
        else:
            return True
    
    varnames_lower = [name.lower() for name in varnames]
    for name in varnames_lower:
        if include_arg_(name):
            ivar, discovered = scoper.search_scope_for_variable(\
              scope,name) # TODO treat implicit here
            argname = name
            if not discovered:
                arg = _intrnl_init_arg(name,"TODO declaration not found","",[],"TODO declaration not found")
                unknown_args.append(arg)
            else:
                arg, lower_bound_args, count_args, macro = _intrnl_create_argument_context(ivar,name,deviceptr_names)
                argname = name.lower().replace("%","_") # TODO
                # modify argument
                if argname in loop_vars: # specific for loop kernels
                    arg["qualifiers"]=[]
                    local_cpu_routine_args.append(arg)
                elif argname in local_vars:
                    arg["qualifiers"]=[]
                    if ivar["rank"] > 0:
                        arg["c_size"] = ivar["total_count"]
                    if "shared" in ivar["qualifiers"]:
                        arg["c_type"] = "__shared__ " + arg["c_type"] 
                    local_cpu_routine_args.append(arg)
                    c_kernel_local_vars.append(arg)
                else:
                    rank = ivar["rank"]
                    if rank > 0: 
                        input_arrays.append({ "name" : name, "rank" : rank })
                        arg["c_size"]    = ""
                        dimensions = "dimension({0})".format(",".join([":"]*rank))
                        # Fortran size expression for allocate
                        f_size = []
                        for i in range(0,rank):
                            f_size.append("{lb}:{lb}+{siz}-1".format(\
                                lb=lower_bound_args[i]["name"],siz=count_args[i]["name"]))
                        local_cpu_routine_args.append(\
                          { "name" : name,
                            "type" : arg["orig_type"],
                            "qualifiers" : ["allocatable",dimensions,"target"],
                            "bounds" : ",".join(f_size),
                            "bytes_per_element" : arg["bytes_per_element"]
                          }\
                        )
                    kernel_args.append(arg)
                    for count_arg in count_args:
                        kernel_args.append(count_arg)
                    for bound_arg in lower_bound_args:
                        kernel_args.append(bound_arg)
                if not macro is None:
                    macros.append(macro)

    # remove unknown arguments that are actually bound variables (<arg>_n<dim> or <arg>_lb<dim>)
    for unknown_kernel_arg in unknown_args:
        append = True
        for kernel_arg in kernel_args:
            if unknown_kernel_arg["name"].lower() == kernel_arg["name"].lower():
                append = False
                break
        if append:
            kernel_args.append(unknown_kernel_arg)

    utils.logging.log_leave_function(LOG_PREFIX,"_intrnl_derive_kernel_arguments")
    return kernel_args, c_kernel_local_vars, macros, input_arrays, local_cpu_routine_args
    
def _intrnl_update_context_from_loop_kernels(loop_kernels,index,hip_context,f_context):
    """
    loop_kernels is a list of STCufloop_kernel objects.
    hip_context, f_context are inout arguments for generating C/Fortran files, respectively.
    """
    utils.logging.log_enter_function(LOG_PREFIX,"_intrnl_update_context_from_loop_kernels")
    
    generate_launcher    = EMIT_KERNEL_LAUNCHER
    generate_cpu_launcher = generate_launcher and EMIT_CPU_IMPLEMENTATION
    
    hip_context["have_reductions"] = False
    for stkernel in loop_kernels:
        parse_result = stkernel.parse_result
        parent_tag   = stkernel.parent.tag()
        scope        = scoper.create_scope(index,parent_tag)

        kernel_args, c_kernel_local_vars, macros, input_arrays, local_cpu_routine_args =\
          _intrnl_derive_kernel_arguments(scope,\
            parse_result.variables_in_body(),\
            parse_result.local_scalars(),\
            parse_result.loop_vars(),\
            True, parse_result.deviceptrs())
        
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
        for arg in kernel_args:
            name  = arg["name"]
            c_type = arg["c_type"]
            cpu_kernel_call_arg_names.append(name)
            is_reduction_var = False
            for op,variables in reductions.items():
                if name.lower() in [var.lower() for var in variables]:
                    # modify argument
                    arg["qualifiers"].remove("value")
                    arg["c_type"] = c_type + "*"
                    # reduction_vars buffer var
                    buffer_name = "_d_" + name
                    var = { "buffer": buffer_name, "name" : name, "type" : c_type, "op" : op }
                    reduction_vars.append(var)
                    # call args
                    kernel_call_arg_names.append(buffer_name)
                    is_reduction_var = True
            if not is_reduction_var:
                kernel_call_arg_names.append(name)
                if type(stkernel) is scanner.STAccLoopKernel:
                    if len(arg["c_size"]):
                        stkernel.append_default_present_var(name)
            hip_context["have_reductions"] |= is_reduction_var
        # C loop kernel
        dimensions  = parse_result.num_dimensions()
        block = _intrnl_convert_dim3(parse_result.num_threads_in_block(),dimensions)
        # TODO more logging
        if not len(block):
            default_block_size = GET_BLOCK_DIMS(kernel_name,dimensions)
            block = _intrnl_convert_dim3(default_block_size,dimensions)
        hip_kernel_dict = {}
        hip_kernel_dict["is_loop_kernel"]          = True
        hip_kernel_dict["modifier"]                = "__global__"
        hip_kernel_dict["return_type"]             = "void"
        hip_kernel_dict["generate_debug_code"]     = EMIT_DEBUG_CODE
        hip_kernel_dict["generate_launcher"]       = generate_launcher 
        hip_kernel_dict["generate_cpu_launcher"]   = generate_cpu_launcher
        
        launch_bounds = GET_LAUNCH_BOUNDS(kernel_name)
        if launch_bounds != None and len(launch_bounds):
            hip_kernel_dict["launch_bounds"]      = "__launch_bounds___({})".format(launch_bounds)
        else:
            hip_kernel_dict["launch_bounds"]      = ""
        hip_kernel_dict["size"]                   = _intrnl_convert_dim3(parse_result.problem_size(),dimensions,do_filter=False)
        hip_kernel_dict["grid"]                   = _intrnl_convert_dim3(parse_result.num_gangs_teams_blocks(),dimensions)
        hip_kernel_dict["block"]                  = block
        hip_kernel_dict["grid_dims"  ]            = [ "{}_grid{}".format(kernel_name,x["dim"])  for x in block ] # grid might not be always defined
        hip_kernel_dict["block_dims"  ]           = [ "{}_block{}".format(kernel_name,x["dim"]) for x in block ]
        hip_kernel_dict["kernel_name"]            = kernel_name
        hip_kernel_dict["macros"]                 = macros
        hip_kernel_dict["c_body"]                 = parse_result.c_str()
        original_snippet = "".join(stkernel.lines())
        if PRETTIFY_EMITTED_FORTRAN_CODE:
            hip_kernel_dict["f_body"]                = utils.fileutils.prettify_f_code(original_snippet)
        else:
            hip_kernel_dict["f_body"]                = original_snippet
        hip_kernel_dict["kernel_args"]               = ["{} {}{}{}".format(a["c_type"],a["name"],a["c_size"],a["c_suffix"]) for a in kernel_args]
        hip_kernel_dict["kernel_call_arg_names"]     = kernel_call_arg_names
        hip_kernel_dict["cpu_kernel_call_arg_names"] = cpu_kernel_call_arg_names
        hip_kernel_dict["reductions"]                = reduction_vars
        hip_kernel_dict["kernel_local_vars"]         = ["{} {}{}".format(a["c_type"],a["name"],a["c_size"]) for a in c_kernel_local_vars]
        hip_kernel_dict["interface_name"]            = kernel_launcher_name
        hip_kernel_dict["interface_comment"]         = "" # kernel_launch_info.c_str()
        hip_kernel_dict["interface_args"]            = hip_kernel_dict["kernel_args"]
        hip_kernel_dict["interface_arg_names"]       = [arg["name"] for arg in kernel_args] # excludes the stream;
        hip_kernel_dict["input_arrays"]              = input_arrays
        #inout_arrays_in_body                        = [name.lower for name in parse_result.inout_arrays_in_body()]
        #hip_kernel_dict["output_arrays"]            = [array for array in input_arrays if array.lower() in inout_arrays_in_body]
        hip_kernel_dict["output_arrays"]          = input_arrays
        hip_context["kernels"].append(hip_kernel_dict)

        if generate_launcher:
            # Fortran interface with automatic derivation of stkernel launch parameters
            f_interface_dict_auto = {}
            f_interface_dict_auto["c_name"]   = kernel_launcher_name + "_auto"
            f_interface_dict_auto["f_name"]   = kernel_launcher_name + "_auto"
            f_interface_dict_auto["type"]     = "subroutine"
            f_interface_dict_auto["args"]     = [
              {"type" : "integer(c_int)", "qualifiers" : ["value", "intent(in)"], "name" : "sharedmem", "c_size" : "" },
              {"type" : "type(c_ptr)"   , "qualifiers" : ["value", "intent(in)"], "name" : "stream",   "c_size": ""},
            ]
            f_interface_dict_auto["args"]    += kernel_args
            f_interface_dict_auto["argnames"] = [arg["name"] for arg in f_interface_dict_auto["args"]]
            
            #######################################################################
            # Feed argument names back to STLoopKernel for host code modification
            #######################################################################
            stkernel.kernel_arg_names = [arg["callarg_name"] for arg in kernel_args] # TODO(refactor): This is where we need to turn things around
            stkernel.grid_f_str       = parse_result.grid_expression_f_str()
            stkernel.block_f_str      = parse_result.block_expression_f_str()
            # TODO use indexer to check if block and dim expressions are actually dim3 types or introduce overloaded make_dim3 interface to hipfort
            stkernel.stream_f_str     = parse_result.stream()    # TODO consistency
            stkernel.sharedmem_f_str  = parse_result.sharedmem() # TODO consistency

            # Fortran interface with manual specification of stkernel launch parameters
            f_interface_dict_manual = copy.deepcopy(f_interface_dict_auto)
            f_interface_dict_manual["c_name"] = kernel_launcher_name
            f_interface_dict_manual["f_name"] = kernel_launcher_name
            f_interface_dict_manual["args"] = [
              {"type" : "type(dim3)", "qualifiers" : ["intent(in)"], "name" : "grid", "c_size": ""},
              {"type" : "type(dim3)", "qualifiers" : ["intent(in)"], "name" : "block", "c_size": ""},
              {"type" : "integer(c_int)", "qualifiers" : ["value", "intent(in)"], "name" : "sharedmem", "c_size" : "" },
              {"type" : "type(c_ptr)"   , "qualifiers" : ["value", "intent(in)"], "name" : "stream",   "c_size": ""},
            ]
            f_interface_dict_manual["args"]    += kernel_args
            f_interface_dict_manual["argnames"] = [arg["name"] for arg in f_interface_dict_manual["args"]]
            f_interface_dict_manual["do_test"]   = False
            
            f_context["interfaces"].append(f_interface_dict_manual)
            f_context["interfaces"].append(f_interface_dict_auto)

            if generate_cpu_launcher: # TODO(refactor): Might disable this at the beginning.
                # External CPU interface
                f_cpu_interface_dict = copy.deepcopy(f_interface_dict_auto)
                f_cpu_interface_dict["f_name"] = "{}_cpu".format(kernel_launcher_name)
                f_cpu_interface_dict["c_name"] = "{}_cpu".format(kernel_launcher_name)
                f_cpu_interface_dict["do_test"] = False

                # Internal CPU routine
                f_cpu_routine_dict = copy.deepcopy(f_interface_dict_auto)
                f_cpu_routine_dict["f_name"] = "{}_cpu1".format(kernel_launcher_name)
                f_cpu_routine_dict["c_name"] = "{}_cpu1".format(kernel_launcher_name)
                
                # rename copied modified args
                for i,val in enumerate(f_cpu_routine_dict["args"]):
                    var_name = val["name"]
                    if val.get("is_array",False):
                        f_cpu_routine_dict["args"][i]["name"] = "d_{}".format(var_name)

                f_cpu_routine_dict["argnames"] = [a["name"] for a in f_cpu_routine_dict["args"]]
                f_cpu_routine_dict["args"]    += local_cpu_routine_args # ordering important
                # add mallocs, memcpys , frees
                prolog = ""
                epilog = "\n"
                for arg in local_cpu_routine_args:
                     if len(arg.get("bounds","")): # is local Fortran array
                       local_array = arg["name"]
                       # device to host
                       prolog += "allocate({var}({bounds}))\n".format(var=local_array,bounds=arg["bounds"])
                       prolog += "CALL hipCheck(hipMemcpy(c_loc({var}),d_{var},{bpe}_8*SIZE({var}),hipMemcpyDeviceToHost))\n".format(var=local_array,bpe=arg["bytes_per_element"])
                       # host to device
                       epilog += "CALL hipCheck(hipMemcpy(d_{var},c_loc({var}),{bpe}_8*SIZE({var}),hipMemcpyHostToDevice))\n".format(var=local_array,bpe=arg["bytes_per_element"])
                       epilog += "deallocate({var})\n".format(var=local_array)
                f_cpu_routine_dict["body"] = prolog + "\n".join(stkernel.code).rstrip() + epilog

                # Add all definitions to context
                f_context["interfaces"].append(f_cpu_interface_dict)
                f_context["routines"].append(f_cpu_routine_dict)
    
    utils.logging.log_leave_function(LOG_PREFIX,"_intrnl_update_context_from_loop_kernels")

# TODO check if this can be combined with other routine
def _intrnl_update_context_from_device_procedures(device_procedures,index,hip_context,f_context):
    """
    device_procedures is a list of STProcedure objects.
    hip_context, f_context are inout arguments for generating C/Fortran files, respectively.
    """
    global EMIT_KERNEL_LAUNCHER
    global EMIT_CPU_IMPLEMENTATION
    global EMIT_DEBUG_CODE

    utils.logging.log_enter_function(LOG_PREFIX,"_intrnl_update_context_from_device_procedures")
    
    for stprocedure in device_procedures:
        iprocedure  = stprocedure.index_record
        is_function = stprocedure.is_function()
        scope       = scoper.create_scope(index,stprocedure.tag())
        
        hip_context["includes"] += _intrnl_create_includes_from_used_modules(iprocedure,index)

        fBody = "\n".join(stprocedure.code)
        parse_result = stprocedure.parse_result
        utils.logging.log_debug3(LOG_PREFIX,"_intrnl_update_context_from_device_procedures","parse result:\n```"+parse_result.c_str().rstrip()+"\n```")

        # TODO: look up functions and subroutines called internally and supply to parse_result before calling c_str()
    
        ## general
        generate_launcher    = EMIT_KERNEL_LAUNCHER and stprocedure.is_kernel_subroutine()
        kernel_name          = iprocedure["name"]
        kernel_launcher_name = "launch_" + kernel_name

        # sort identifiers: put dummy args first
        varnames   = [scoper.create_index_search_tag_for_variable(varexpr) for varexpr in parse_result.variables_in_body()]
        local_vars = [varname for varname in varnames if varname not in iprocedure["dummy_args"]]
        ordered_varnames = iprocedure["dummy_args"] + local_vars

        # TODO also check 'used' variables from other modules; should be in scope
        # TODO also add implicit variables; should be in scope

        kernel_args, c_kernel_local_vars, macros, input_arrays, local_cpu_routine_args =\
          _intrnl_derive_kernel_arguments(scope,\
            ordered_varnames,local_vars,[],\
            False,deviceptr_names=[])

        # C routine and C stprocedure launcher
        hip_kernel_dict = {}
        launch_bounds = GET_LAUNCH_BOUNDS(kernel_name)
        if launch_bounds != None and len(launch_bounds) and stprocedure.is_kernel_subroutine():
            hip_kernel_dict["launch_bounds"]     = "__launch_bounds___({})".format(launch_bounds)
        else:
            hip_kernel_dict["launch_bounds"]     = ""
        hip_kernel_dict["generate_debug_code"]   = EMIT_DEBUG_CODE
        hip_kernel_dict["generate_launcher"]     = generate_launcher
        hip_kernel_dict["generate_cpu_launcher"] = False
        hip_kernel_dict["modifier"]              = "__global__" if stprocedure.is_kernel_subroutine() else "__device__"
        hip_kernel_dict["return_type"]           = stprocedure.c_result_type
        hip_kernel_dict["is_loop_kernel"]        = False
        hip_kernel_dict["kernel_name"]           = kernel_name
        hip_kernel_dict["macros"]                = macros
        hip_kernel_dict["c_body"]                = parse_result.c_str()
        hip_kernel_dict["f_body"]                = "".join(stprocedure.lines()).rstrip("\n")
        hip_kernel_dict["kernel_args"] = []
        # device procedures take all C args as reference or pointer
        # kernel proceduers take all C args as value or (device) pointer
        for arg in kernel_args:
            c_type = arg["c_type"]
            if not stprocedure.is_kernel_subroutine() and not arg["is_array"]:
                c_type += "&"
            hip_kernel_dict["kernel_args"].append(c_type + " " + arg["name"])
        hip_kernel_dict["kernel_local_vars"]       = ["{0} {1}{2}{3}".format(a["c_type"],a["name"],a["c_size"],"= " + a["c_suffix"] if len(a["c_suffix"]) else "") for a in c_kernel_local_vars]
        hip_kernel_dict["interface_name"]         = kernel_launcher_name
        hip_kernel_dict["interface_args"]         = hip_kernel_dict["kernel_args"]
        hip_kernel_dict["interface_comment"]      = ""
        hip_kernel_dict["interface_arg_names"]     = [arg["name"] for arg in kernel_args]
        hip_kernel_dict["input_arrays"]           = input_arrays
        #inout_arrays_in_body                   = [name.lower for name in parse_result.inout_arrays_in_body()]
        #hip_kernel_dict["output_arrays"]       = [array for array in input_arrays if array.lower() in inout_arrays_in_body]
        hip_kernel_dict["output_arrays"]          = input_arrays
        hip_kernel_dict["kernel_call_arg_names"]    = hip_kernel_dict["interface_arg_names"] # TODO(05/12/21): Normally this information must be passed to other kernels
        hip_kernel_dict["cpu_kernel_call_arg_names"] = hip_kernel_dict["interface_arg_names"] 
        hip_kernel_dict["reductions"]            = []
        hip_context["kernels"].append(hip_kernel_dict)

        if generate_launcher:
            # Fortran interface with manual specification of kernel launch parameters
            f_interface_dict_manual = {}
            f_interface_dict_manual["c_name"]       = kernel_launcher_name
            f_interface_dict_manual["f_name"]       = kernel_launcher_name
            f_interface_dict_manual["test_comment"] = ["Fortran implementation:"] + stprocedure.code
            f_interface_dict_manual["type"]         = "subroutine"
            f_interface_dict_manual["args"]         = [
                {"type" : "type(dim3)", "qualifiers" : ["intent(in)"], "name" : "grid"},
                {"type" : "type(dim3)", "qualifiers" : ["intent(in)"], "name" : "block"},
                {"type" : "integer(c_int)", "qualifiers" : ["value", "intent(in)"], "name" : "sharedmem"},
                {"type" : "type(c_ptr)", "qualifiers" : ["value", "intent(in)"], "name" : "stream"},
            ]
            f_interface_dict_manual["args"]    += kernel_args
            f_interface_dict_manual["argnames"] = [arg["name"] for arg in f_interface_dict_manual["args"]]
            f_interface_dict_manual["do_test"]   = True
            f_context["interfaces"].append(f_interface_dict_manual)
    
    utils.logging.log_enter_function(LOG_PREFIX,"_intrnl_update_context_from_device_procedures")

def _intrnl_write_file(outfile_path,kind,content):
    utils.logging.log_enter_function(LOG_PREFIX,"_intrnl_write_file")
    
    with open(outfile_path,"w") as outfile:
        outfile.write(content)
        msg = "created {}: ".format(kind).ljust(40) + outfile_path
        utils.logging.log_info(LOG_PREFIX,"_intrnl_write_file",msg)
    
    utils.logging.log_leave_function(LOG_PREFIX,"_intrnl_write_file")


def _intrnl_create_includes_from_used_modules(index_record,index):
    """Create include statement for a module's/subprogram's used modules that are present in the index."""
    used_modules  = [irecord["name"] for irecord in index_record["used_modules"]]
    includes     = []
    for iuse in index:
        if iuse["name"] in used_modules:
            includes.append(iuse["name"] + HIP_FILE_EXT)
    return includes
# API
exec(open(os.path.join(fort2hip_dir,"fort2hip_gpufort_sources.py.in")).read())

def generate_hip_files(stree,index,kernels_to_convert_to_hip,translation_source_path,generate_code):
    """
    :param stree:        [inout] the scanner tree holds nodes that store the Fortran code lines of the kernels
    :param generate_code: generate code or just feed kernel signature information
                         back to the scanner tree.
    :note The signatures of the identified kernels must be fed back to the 
          scanner tree even when no kernel files are written.
    """
    global FORTRAN_MODULE_PREAMBLE
    global PRETTIFY_EMITTED_C_CODE
    global PRETTIFY_EMITTED_FORTRAN_CODE
    global CLANG_FORMAT_STYLE
    global FORTRAN_MODULE_FILE_EXT
    global HIP_FILE_EXT    
    global FORTRAN_MODULE_SUFFIX

    utils.logging.log_enter_function(LOG_PREFIX,"generate_hip_files",\
      {"kernels_to_convert_to_hip":" ".join(kernels_to_convert_to_hip),\
       "translation_source_path": translation_source_path,\
       "generate_code":generate_code})
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
    def device_procedure_filter_(child):
        return type(child) is scanner.STProcedure and\
          child.must_be_available_on_device() and select_(child)

    fortran_module_filepath = None
    main_hip_filepath       = None
    output_dir             = os.path.dirname(translation_source_path)
   
    have_reductions     = False
    hip_module_filenames = []
    fortran_modules     = []
    program_or_modules = stree.find_all(filter=lambda child: type(child) in [scanner.STProgram,scanner.STModule], recursively=False)
    for stmodule in program_or_modules:
        # file names & paths
        module_name         = stmodule.name.lower()
        hip_module_filename = module_name + HIP_FILE_EXT
        hip_module_filenames.append(hip_module_filename)
        hip_module_filepath = output_dir+"/"+hip_module_filename
        guard               = "__"+hip_module_filename.replace(".","_").replace("-","_").upper()+"__"
        # extract kernels
        loop_kernels      = stmodule.find_all(filter=loop_kernel_filter_, recursively=True)
        device_procedures = stmodule.find_all(filter=device_procedure_filter_, recursively=True)
        # TODO: Also extract derived types
        # derivedtypes = ....
        
        # TODO handle includes
        imodule = next((ilinemap for ilinemap in index if ilinemap["name"] == module_name),None)
        if imodule == None:
            utils.logging.log_error(LOG_PREFIX,"generate_hip_files","could not find linemap for module '{}'.".format(module_name))
            sys.exit() # TODO add error code

        includes = _intrnl_create_includes_from_used_modules(imodule,index)
        itypes   = _intrnl_search_derived_types(imodule)
        if len(loop_kernels) or len(device_procedures) or len(itypes):
            utils.logging.log_debug2(LOG_PREFIX,"generate_hip_files",\
              "detected loop kernels: {}; detected device subprograms {}".format(\
              len(loop_kernels),len(device_procedures)))

            # Context for HIP implementation
            hip_context = {}
            hip_context["guard"]    = guard 
            hip_context["includes"] = [ "hip/hip_runtime.h", "hip/hip_complex.h" ] + includes
            hip_context["kernels"]  = []
            hip_context["types"]    = []
            
            # Context for Fortran interface/implementation
            f_context = {}
            f_context["name"]     = module_name + FORTRAN_MODULE_SUFFIX
            f_context["preamble"] = ""
            f_context["used"]     = ["hipfort"]
            if EMIT_CPU_IMPLEMENTATION:
                f_context["used"].append("hipfort_check")

            f_context["interfaces"] = []
            f_context["routines"]   = []
            f_context["types"]      = []
            
            _intrnl_update_context_from_derived_types(itypes,hip_context,f_context)
            _intrnl_update_context_from_loop_kernels(loop_kernels,index,hip_context,f_context)
            _intrnl_update_context_from_device_procedures(device_procedures,index,hip_context,f_context)

            if generate_code:
                have_reductions = have_reductions or hip_context["have_reductions"]

                _intrnl_write_file(\
                   hip_module_filepath,"HIP C++ implementation file",\
                   model.HipImplementationModel().generate_code(hip_context))
                if PRETTIFY_EMITTED_C_CODE:
                    utils.fileutils.prettify_c_file(hip_module_filepath,CLANG_FORMAT_STYLE)
                if len(f_context["interfaces"]):
                   fortran_modules.append(\
                     model.InterfaceModuleModel().generate_code(f_context))
        else:
            content = "\n".join(["#include \"{}\"".format(filename) for filename in includes])
            if len(content):
                content = "#ifndef {0}\n#define {0}\n{1}\n#endif // {0}".format(
                  guard,content)
            _intrnl_write_file(\
               hip_module_filepath,"HIP C++ implementation file",content)

    if generate_code:
        # main HIP file
        main_hip_filepath = translation_source_path + HIP_FILE_EXT
        content = "\n".join(["#include \"{}\"".format(filename) for filename in hip_module_filenames])
        _intrnl_write_file(main_hip_filepath,"main HIP C++ file",content)

        # Fortran module file
        if len(fortran_modules):
            fortran_module_filepath = translation_source_path + FORTRAN_MODULE_FILE_EXT
            content               = "\n".join(fortran_modules)
            if len(FORTRAN_MODULE_PREAMBLE):
                content = FORTRAN_MODULE_PREAMBLE + "\n" + content
            _intrnl_write_file(fortran_module_filepath,"interface/testing module",content)
            if PRETTIFY_EMITTED_FORTRAN_CODE:
                utils.fileutils.prettify_f_file(fortran_module_filepath)
    
    utils.logging.log_leave_function(LOG_PREFIX,"generate_hip_files")
    
    return fortran_module_filepath, main_hip_filepath
