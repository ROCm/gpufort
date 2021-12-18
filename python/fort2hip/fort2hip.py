
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

# TODO refactor - derived directly from index
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


def render_derived_types_f03(itypes,used_modules):
    derived_type_snippets = []
    procedure_snippets    = []

    fort2hip.model.render_derived_types_f03(derived_types,interop_suffix="_interop")
    fort2hip.model.render_derived_type_copy_scalars_routines_f03(derived_types,interop_suffix="_interop",used_modules=[])
    fort2hip.model.render_derived_type_size_bytes_routines_f03(derived_types,interop_suffix="_interop",used_modules=[])
    fort2hip.model.render_derived_type_copy_array_member_routines_f03(derived_types,interop_suffix="_interop",orig_var="orig_type",interop_var="interop_type",used_modules=[])
   
    return derived_type_snippets, procedure_snippets

def traverse_scanner_tree(stree,index):
    cpp_file_contexts = []
    f03_file_contexts = []
    #
    def traverse_node_(stnode)
        """Traverse the scanner tree and emit code generation context.
        """
        nonlocal index
        nonlocal cpp_file_contexts
        nonlocal f03_file_contexts
        #
        def fort2hip_module_name_(stnode):
            global FORTRAN_SUFFIX
            return "{}{}".format(stnode.tag().replace(":","_"),FORTRAN_MODULE_SUFFIX)
        def parent_fort2hip_modules_(stnode):
            if stnode.parent != None and type(stnode.parent) != scanner.STRoot
                return [{ "name": fort2hip_module_name_(stnode.parent.name.lower), "only": [] }]
            else:
                return []
        def used_modules_(stnode,inode):
           used_modules = [{"name": mod,"only:"[]} for mod in ["gpufort_array"]
           used_modules += inode["used_modules"] # local modules
           used_modules += parent_fort2hip_modules_(stnode)  
           return used_modules
        # 
        if type(stnode) == scanner.STRoot:
            for stchildnode in stnode.children:
                traverse_node(stchildnode,index)
        elif type(stnode) in [scanner.STProgram,scanner.STModule,STProcedure]:
            stnode_name                 = stnode.name.lower()
            stnode_fort2hip_module_name = fort2hip_module_name_(stnode)
            inode = next((irecord for irecord in index if irecord["name"] == stnode_name),None)
            if inode == None:
                utils.logging.log_error(LOG_PREFIX,\
                                        "traverse_scanner_tree",\
                                        "could not find index record for scanner tree node '{}'.".format(stnode_name))
                sys.exit() # TODO add error code
           itypes_local = inode["types"] # local types
           procedure_used_modules = used_modules_(stnode,inode) # compute_unit
           # derived_type_snippets = render_types(itypes) 
           module_used_modules = compute_unit_used_modules if (type(stnode) == STModule) else []
           for stchildnode in stnode.children:
               traverse_node(stchildnode)


def traverse_scanner_tree(stree,index,kernels_to_convert_to_hip=["*"])
    """Traverse the scanner tree."""
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
    #  TODO distinguish between device routines and global routines
    def device_procedure_filter_(child):
        return type(child) is scanner.STProcedure and\
          child.must_be_available_on_device() and select_(child)
    
         

def render_interopable_types(stcontainer,index):
    pass

def render_device_function()
    pass

def render_loop_kernel(stkernel,index):
    pass
