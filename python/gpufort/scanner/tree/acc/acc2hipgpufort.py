# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import textwrap

from gpufort import util
from gpufort import translator
from gpufort import indexer

from ... import opts

from .. import nodes
from .. import backends

from . import accbackends
from . import accnodes

## DIRECTIVES
HIP_GPUFORT_RT_ACC_PRESENT = "call gpufort_acc_ignore(gpufort_acc_present({var}{asyncr}{alloc}))\n"
# update
HIP_GPUFORT_RT_ACC_UPDATE = "call gpufort_acc_update_{kind}({var}{asyncr})\n"
# wait
HIP_GPUFORT_RT_ACC_WAIT = "call gpufort_acc_wait({queue}{asyncr})\n"
# init shutdown
HIP_GPUFORT_RT_ACC_INIT = "call gpufort_acc_init()\n"
HIP_GPUFORT_RT_ACC_SHUTDOWN = "call gpufort_acc_shutdown()\n"
# regions
HIP_GPUFORT_RT_ACC_ENTER_REGION = "call gpufort_acc_enter_region({region_kind})\n"
HIP_GPUFORT_RT_ACC_EXIT_REGION = "call gpufort_acc_exit_region({region_kind})\n"

HIP_GPUFORT_RT_CLAUSES_OMP2ACC = {
    "alloc": "create",
    "to": "copyin",
    "tofrom": "copy"
}

def dev_var_name(var):
    #tokens = var.split("%")
    #tokens[-1] = ACC_DEV_PREFIX+tokens[-1]+ACC_DEV_SUFFIX
    #return "%".join(tokens)
    result = var.replace("%", "_")
    result = result.replace("(", "$")
    result = result.replace(")", "$")
    result = "".join(c for c in result if c.isalnum() or c in "_$")
    result = result.replace("$$", "")
    result = result.replace("$", "_")
    return opts.acc_dev_prefix + result + opts.acc_dev_suffix

@util.logging.log_entry_and_exit(opts.log_prefix)
def add_implicit_region(stcontainer):
    last_decl_list_node = stcontainer.last_entry_in_decl_list()
    indent = last_decl_list_node.first_line_indent()
    last_decl_list_node.add_to_epilog(textwrap.indent(HIP_GPUFORT_RT_ACC_ENTER_REGION,format(\
        region_kind="implicit_region=.true."),indent))
    for stendorreturn in stcontainer.return_or_end_statements():
        stendorreturn.add_to_prolog(textwrap.indent(HIP_GPUFORT_RT_ACC_EXIT_REGION.format(\
            region_kind="implicit_region=.true."),indent))

# `In Fortran, if a variable or array of composite type is specified, all the members of that derived
# type are allocated and copied, as appropriate. If any member has the allocatable or
# pointer attribute, the data accessed through that member are not copied.`
# => Copy fixed size arrays per default and require manual copy of other data
# => Emit error if a non-allocatable, non-pointer var should be mapped
#for parent_type in util.parsing.derived_type_parents(var_expr.lower()):
#    parent_already_mapped = parent_already_mapped or parent_type in mappings.keys()
#    if parent_already_mapped: break;
#if not parent_already_mapped:
# => 

# TODO make ivar pair of expr and ivar


def _mapping(clause_kind,tavar,**kwargs):
    asyncr,_   = util.kwargs.get_value("asyncr","",**kwargs)
    finalize,_ = util.kwargs.get_value("finalize","",**kwargs)
   
    clauses = [
      "create","no_create","delete","copyin",
      "copy","copyout","present",
    ]
    
    if clause.kind in clauses:
        copy_variants = ["copy","copyin","copyout"]
        clauses_with_async = (["present"]
                              + ["present_or_{}".format(v) for v in copy_variants] 
                              + copy_variants)
        clauses_with_finalize = ["delete"] 

        tokens = list(tokens1)
        tokens.append(var_expr)
        if len(alloc) clause.kind in clauses_with_alloc:
            tokens += [",",alloc]
        if len(asyncr) and clause.kind in clauses_with_async:
            tokens += [",",asyncr]
        if len(finalize) and clause.kind in clauses_with_finalize:
            tokens += [",",finalize]
        tokens.append(")") 
        return "".join(tokens)
    else:
        raise util.parsing.SyntaxError("clause not supported") 

def kernel_args_to_acc_mappings(directive,tavars,present_by_default,callback,**kwargs):
    """Derive mappings for array and (not yet) derived type variables.
   
    :Implemented:
    
    * Arrays of basic type are looked up and put into gpufort_array_wrap_device_ptr
      * If scalar is part of an array, the whole array is mapped.
    
    :Not implemented yet:
      * Special treatment for scalar derived types
        * IGNORED for the time being
      * Special treatment of array of derived type
        * IGNORED for the time being
        * need special routine for this pre-generated for derived types
        * different treatment if array is part of parent derived type
          and parent derived type has been mapped
    
    :param bool present_by_default: If unmapped vars are present by default or not.
    :note: Current implementation does not support mapping of scalars/arrays of user-defined type.
    :raise util.parsing.LimitationError: 
    :raise util.parsing.LimitationError: 
    """
    #
    #translator.tree.grammar.acc_mapping_clause.scanString():
    mappings = {}
    try:
        accclauses = [res[0][0] for res in translator.tree.grammar.acc_mapping_clause.scanString(directive)]
    except Exception:
        raise util.parsing.SyntaxError("could not parse OpenACC clauses")
    for tavar in tavars:
        if tavar["rank"] > 0:
            explicitly_mapped = False
            for clause in accclauses: 
                tokens1 = ["gpufort_acc_",ttaccclause.kind,"("]
                    for var_expr in clause.var_expressions():
                        var_tag = util.parsing.strip_array_indexing(var_expr.lower())
                        if tavar["expr"] == var_tag:
                            if "%" in var_tag:
                                raise util.parsing.LimitationError("mapping of derived type members not supported (yet)")
                            else:
                                mappings[tavar["expr"]] = callback(clause.kind,tavar,**kwargs)
                                explicitly_mapped = True
                                break
                
                if explicitly_mapped: break
            if not explicitly_mapped and present_by_default
                if "%" in tavar["expr"] or tavar["f_type"]=="type":
                    raise util.parsing.LimitationError("mapping of derived types and their members not supported (yet)")
                else:
                    mappings[tavar["expr"]] = callback("present_or_copy",tavar,**kwargs)
            elif not explicitly_mapped:       
                return util.parsing.SyntaxError("no mapping specified for expression: {}".format(tavar["expr"]))
    return mappings    

def derive_kernel_arguments(directive,ivars,**kwargs):
    """:return a list of arguments given the directives
    """
    # TODO how to handle case where a variable is part of a mapping
    # but not present in the body?
    # TODO have a preproc step where derived type members are kicked
    # out of the argument list if the parent struct is mapped.
    # TODO Simplify ACC parsing, no need to use pyparsing
    kernel_args_to_acc_mappings 

class Acc2HipGpufortRT(accbackends.AccBackendBase):
    # clauses
    def _handle_async(self, queue=None, prefix=",asyncr="):
        """:return: Empty string if no queue was found
        :rtype: str
        """
        result = ""
        if queue is None:
            for parse_result in translator.tree.grammar.acc_clause_async.searchString(
                    queue, 1):
                #print(parse_result)
                # TODO find ...
                result = parse_result[0].queue()
        if len(result):
            result = prefix + result
        return result

    def _handle_finalize(self):
        """:return: If a finalize clause is present
        :rtype: bool
        """
        return len(
            translator.tree.grammar.acc_clause_finalize.searchString(
                self.first_statement(), 1))

    def _handle_mapping_clauses(self):
        """
        """
        result = ""
        temp_vars = set()
        #
        for parse_result in translator.tree.grammar.acc_mapping_clause.scanString(
                self._stnode.first_statement()):
            clause = parse_result[0][0]
            if clause.kind in MAPPING_CLAUSE_2_TEMPLATE_MAP:
                var_names = clause.var_names()
                var_expressions = clause.var_expressions()
                for i, var_expr in enumerate(var_expressions):
                    deviceptr = dev_var_name(var_names[i])
                    template = MAPPING_CLAUSE_2_TEMPLATE_MAP[clause.kind]
                    result += template.format(var=var_expr,dev_var=deviceptr,\
                        asyncr=self._handle_async(),alloc="",finalize="")
                    temp_vars.add(deviceptr)
        return result, len(result), temp_vars

    def _handle_update(self):
        """Emits a acc_clause_update command for every variable in the list
        """
        result = ""
        # update host
        for parse_result in translator.tree.grammar.acc_mapping_clause.scanString(
                self._stnode.first_statement()):
            clause = parse_result[0][0]
            if clause.kind in ["self", "host", "device"]:
                clause_kind = "host" if clause.kind == "self" else clause.kind
                var_expressions = clause.var_expressions()
                for i, var_expr in enumerate(var_expressions):
                    result += HIP_GPUFORT_RT_ACC_UPDATE.format(
                        var=var_expr,
                        asyncr=self._handle_async(),
                        kind=clause_kind)
        return result, len(result)

    def _handle_wait(self):
        """
        Emits a acc_clause_wait command for every variable in the list
        """
        result = ""
        temp_vars = set()
        # wait
        template = HIP_GPUFORT_RT_ACC_WAIT
        for parse_result in translator.tree.grammar.acc_clause_wait.scanString(
                self._stnode.first_statement()):
            queue_list = []
            asyncr_list = []
            for rvalue in parse_result[0][0]: # queue ids
                queue_list.append(rvalue.var_name())
            for rvalue in parse_result[0][1]: # asyncr queue ids
                asyncr_list.append(rvalue.var_name())
            queue = ""
            asyncr = ""
            if len(queue_list):
                queue = ",[{}]".format(",".join(queue_list))
            if len(asyncr_list):
                asyncr = ",[{}]".format(",".join(asyncr_list))
            result += template.format(
                                      queue=queue,
                                      asyncr=asyncr)
        return result, len(result), temp_vars

    def _handle_if(self):
        """
        :return: Empty string if no if was found
        :rtype: str
        """
        condition = ""
        for parse_result in translator.tree.grammar.acc_clause_if.searchString(
                self._stnode.first_statement(), 1):
            condition = parse_result[0].condition()
        return condition

    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[],
                  handle_if=True):
        """
        :param line: An excerpt from a Fortran file, possibly multiple lines
        :type line: str
        :return: If the text was changed at all
        :rtype: bool
        """
        result = ""
        all_temp_vars = set()

        ## Init
        stnode = self._stnode
        if stnode.is_directive(["acc","init"]):
            result += HIP_GPUFORT_RT_ACC_INIT
        elif stnode.is_directive(["acc","update"]):
            partial_result, transformed = self._handle_update()
            if transformed:
                result += partial_result
        elif stnode.is_directive(["acc","shutdown"]):
            result += HIP_GPUFORT_RT_ACC_SHUTDOWN
        else:
            ## Enter region commands must come first
            emit_enter_region = stnode.is_directive(["acc","enter","data"])
            if emit_enter_region:
                region_kind = "unstructured=.true."
            else:
                region_kind = ""
            emit_enter_region = (emit_enter_region
                                 or stnode.is_directive(["acc","data"])
                                 or stnode.is_directive(["acc","parallel"])
                                 or stnode.is_directive(["acc","parallel","loop"])
                                 or stnode.is_directive(["acc","kernels"])
                                 or stnode.is_directive(["acc","kernels","loop"]))
            
            if emit_enter_region:
                result += HIP_GPUFORT_RT_ACC_ENTER_REGION.format(
                    region_kind=region_kind)

            ## Other directives/clauses
            partial_result, transformed, temp_vars = self._handle_mapping_clauses(
            )
            if transformed:
                result += partial_result
                all_temp_vars.update(temp_vars)

            ## wait
            partial_result, transformed, _ = self._handle_wait()
            if transformed:
                result += partial_result

            ## Exit region commands must come last
            emit_exit_region = stnode.is_directive(["acc","exit","data"])
            if emit_exit_region:
                region_kind = "unstructured=.true."
            else:
                region_kind = ""
            emit_exit_region = (emit_exit_region 
                               or stnode.is_directive(["acc","end","data"])
                               or (stnode.is_directive(["acc","end","kernels"])
                                  and not stnode.is_directive(["acc","end","kernels","loop"]))
                               or (stnode.is_directive(["acc","end","parallel"])
                                  and not stnode.is_directive(["acc","end","parallel","loop"])))
            if emit_exit_region:
                result += HIP_GPUFORT_RT_ACC_EXIT_REGION.format(
                    region_kind=region_kind)

        # _handle if
        condition = self._handle_if()
        if len(condition) and handle_if:
            result = "if ( {condition} ) then\n{result}\n endif".format(\
                    condition=condition, result=result.rstrip("\n"))

        # introduce the new variables
        stnode.parent.append_vars_to_decl_list(all_temp_vars)

        indent = stnode.first_line_indent()
        return textwrap.indent(result,indent), len(result)

class AccLoopNest2HipGpufortRT(Acc2HipGpufortRT):

    def _handle_default(self):
        """
        Emits a acc_clause_present command for every variable in the list
        """
        stloopnest = self._stnode
        ttloopnest = stloopnest.parse_result
        unmapped_arrays = ttloopnest.all_unmapped_arrays()

        result = ""
        temp_vars = []
        "{dev_var} = gpufort_acc_present({var}{asyncr}{alloc})\n"
        template = HIP_GPUFORT_RT_ACC_PRESENT
        default_clauses = translator.tree.grammar.acc_clause_default.searchString(
            stloopnest.first_statement(), 1)
        if len(default_clauses):
            value = str(default_clauses[0][0][0]).lower()
            #print(value)
            if value == "present":
                for var_expr in unmapped_arrays:
                    deviceptr = dev_var_name(var_expr)
                    temp_vars.append(deviceptr)
                    result += template.format(
                                              var=var_expr,
                                              asyncr="",
                                              alloc="",
                                              dev_var=deviceptr)
            elif value == "none" and len(unmapped_arrays):
                util.logging.log_warning(
                    LOG_PREFIX, "AccLoopNest2HipGpufortRT._handle_default",
                    "'default(none)' specified but no map for the following variables found: "
                    .format(", ".join(unmapped_arrays)))
        else: # default strategy: present_or_copy
            for var_expr in unmapped_arrays:
                deviceptr = dev_var_name(var_expr)
                if not deviceptr in temp_vars:
                    temp_vars.append(deviceptr)
                    result += template.format(
                        
                        var=var_expr,
                        asyncr="",
                        alloc=",or=gpufort_acc_event_copy",
                        dev_var=deviceptr)

        return result, len(result), temp_vars

    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[]):
        result = ""
        stloopnest = self._stnode
        ttloopnest = stloopnest.parse_result
        arrays_in_body = ttloopnest.arrays_in_body()
        if (stloopnest.is_directive(["acc","parallel","loop"])
           or stloopnest.is_directive(["acc","kernels","loop"])):
            result_directive, _ = Acc2HipGpufortRT.transform(
                self,
                joined_lines,
                joined_statements,
                statements_fully_cover_lines,
                index,
                handle_if=False)
            result = textwrap.dedent(result_directive)
        partial_result, transformed, temp_vars = self._handle_default()
        if transformed:
            stloopnest.parent.append_vars_to_decl_list(temp_vars)
        # from scanner/fort2x via translator.analysis # TODO revise
        #for ivar in stloopnest.kernel_args_ivars():  
        #    # check if ivar is listed in the clauses
        #    # if so ,check if it is a array, type, or scalar
        #    # 
        ## 
        partial_result, _ = nodes.STLoopNest.transform(
                                               stloopnest, joined_lines, joined_statements,
                                               statements_fully_cover_lines, index)
        partial_result = textwrap.dedent(partial_result)
        result = "\n".join([result.rstrip("\n"), partial_result])

        # handle default
        self._handle_default()
        # add wait call if necessary
        queue = self._handle_async(None, "")
        if not len(queue):
            result = result.rstrip(
                "\n") + "\n" + HIP_GPUFORT_RT_ACC_WAIT.format(
                    queue=queue, asyncr="")
        #if stloopnest.is_directive(["acc","parallel","loop"])) or stloopnest.is_directive(["acc","kernels","loop"])):
        if (stloopnest.is_directive(["acc","kernels","loop"]) or
           stloopnest.is_directive(["acc","parallel","loop"])):
            result = result.rstrip(
                "\n") + "\n" + HIP_GPUFORT_RT_ACC_EXIT_REGION.format(
                    region_kind="")
        # wrap in ifdef if necessary
        condition = self._handle_if()
        if len(condition):
            result = "if ( {condition} ) then\n{result}\nelse\n {original}\n endif".format(\
                    condition=condition, result=result.rstrip("\n"), original="".join(stloopnest._lines).rstrip("\n"))
        indent = stloopnest.first_line_indent()
        return textwrap.indent(result,indent), len(result)


def AllocateHipGpufortRT(stallocate, joined_statements, index):
    stcontainer = stallocate.parent
    parent_tag = stcontainer.tag()
    scope = indexer.scope.create_scope(index, parent_tag)
    scope_vars = scope["variables"]
    indent = stallocate.first_line_indent()
    local_var_names, dummy_arg_names = stcontainer.local_and_dummy_var_names(
        index)
    acc_present_calls = []
    temp_vars = []
    implicit_region = False
    for var in stallocate.variable_names:
        ivar = indexer.scope.search_index_for_var(index, parent_tag, var)
        host_var = ivar["name"]
        dev_var = dev_var_name(host_var)
        is_local_var = host_var in local_var_names
        is_arg = host_var in dummy_arg_names
        is_used_module_var = not is_local_var and not is_arg
        is_allocatable_or_pointer = "allocatable" in ivar["qualifiers"] or\
                             "pointer" in ivar["qualifiers"]
        assert is_allocatable_or_pointer
        module_var = ",module_var=.true." if is_used_module_var else ""
        if not is_used_module_var:
            implicit_region = True
        if ivar["declare_on_target"] in HIP_GPUFORT_RT_CLAUSES_OMP2ACC.keys():
            map_kind = HIP_GPUFORT_RT_CLAUSES_OMP2ACC[
                ivar["declare_on_target"]]
            acc_present_calls.append(HIP_GPUFORT_RT_ACC_PRESENT.format(\
                var=host_var,asyncr="",\
                alloc=module_var+",or=gpufort_acc_event_"+map_kind,dev_var=dev_var))
            temp_vars.append(dev_var)
    if len(acc_present_calls):
        stcontainer.append_vars_to_decl_list(temp_vars)
        if implicit_region:
            add_implicit_region(stcontainer)
        for line in acc_present_calls:
            stallocate.add_to_epilog(textwrap.indent(line,indent))
    return joined_statements, False


def DeallocateHipGpufortRT(stdeallocate, joined_statements, index):
    stcontainer = stdeallocate.parent
    parent_tag = stcontainer.tag()
    scope = indexer.scope.create_scope(index, parent_tag)
    scope_vars = scope["variables"]
    indent = stdeallocate.first_line_indent()
    local_var_names, dummy_arg_names = stcontainer.local_and_dummy_var_names(
        index)
    acc_delete_calls = []
    for var in stdeallocate.variable_names:
        ivar = indexer.scope.search_index_for_var(index, parent_tag, var)
        host_var = ivar["name"]
        dev_var = dev_var_name(host_var)
        is_local_var = host_var in local_var_names
        is_arg = host_var in dummy_arg_names
        is_used_module_var = not is_local_var and not is_arg
        is_allocatable_or_pointer = "allocatable" in ivar["qualifiers"] or\
                             "pointer" in ivar["qualifiers"]
        assert is_allocatable_or_pointer
        module_var = ",module_var=.true." if is_used_module_var else ""
        if ivar["declare_on_target"] in ["alloc", "to", "tofrom"]:
            acc_delete_calls.append(HIP_GPUFORT_RT_ACC_DELETE.format(\
              var=host_var,asyncr="",finalize="",\
              alloc=module_var,dev_var=dev_var))
    for line in acc_delete_calls:
        stdeallocate.add_to_prolog(textwrap.indent(line,indent))
    return joined_statements, False


def Acc2HipGpufortRTPostprocess(stree, index):
    """:param stree: the full scanner tree
       :param staccdirectives: All acc directive tree accnodes."""
    accbackends.add_runtime_module_use_statements(stree,"gpufort_acc_runtime")

    # TODO check if there is any acc used in the
    # construct at all
    # TODO handle directly via directives; only variables occuring
    # in directives need to be available on device
    containers = stree.find_all(\
      lambda node: type(node) in [nodes.STProgram,nodes.STProcedure],
      recursively=True)
    for stcontainer in containers:
        last_decl_list_node = stcontainer.last_entry_in_decl_list()
        indent = last_decl_list_node.first_line_indent()
        scope = indexer.scope.create_scope(index, stcontainer.tag())
        scope_vars = scope["variables"]
        local_var_names, dummy_arg_names = stcontainer.local_and_dummy_var_names(
            index)
        # TODO also process type members
        acc_present_calls = ""
        temp_vars = []
        implicit_region = False
        for ivar in scope_vars:
            host_var = ivar["name"]
            dev_var = dev_var_name(host_var)
            is_local_var = host_var in local_var_names
            is_arg = host_var in dummy_arg_names
            is_used_module_var = not is_local_var and not is_arg
            is_allocatable = "allocatable" in ivar["qualifiers"]
            is_pointer = "pointer" in ivar["qualifiers"]
            if not is_allocatable:
                if not is_used_module_var:
                    implicit_region = True
                module_var = ",module_var=.true." if is_used_module_var else ""
                if is_pointer:
                    cond = "if (associated({var})) "
                    acc_present_template = (
                        cond
                        + HIP_GPUFORT_RT_ACC_PRESENT.replace("", ""))
                else:
                    acc_present_template = HIP_GPUFORT_RT_ACC_PRESENT
                # find return and end, emit 1 new implicit region for all
                if ivar["declare_on_target"] in HIP_GPUFORT_RT_CLAUSES_OMP2ACC.keys(
                ):
                    map_kind = HIP_GPUFORT_RT_CLAUSES_OMP2ACC[
                        ivar["declare_on_target"]]
                    acc_present_calls += acc_present_template.format(\
                        var=host_var,asyncr="",\
                        alloc=module_var+",or=gpufort_acc_event_"+map_kind,dev_var=dev_var)
                    temp_vars.append(dev_var)
        if len(acc_present_calls):
            stcontainer.append_vars_to_decl_list(temp_vars)
            if implicit_region:
                add_implicit_region(stcontainer)
            last_decl_list_node.add_to_epilog(textwrap.indent(acc_present_calls,indent))

dest_dialects = ["hipgpufort"]
accnodes.STAccDirective.register_backend(dest_dialects,Acc2HipGpufortRT()) # instance
accnodes.STAccLoopNest.register_backend(dest_dialects,AccLoopNest2HipGpufortRT())

nodes.STAllocate.register_backend("acc", dest_dialects, AllocateHipGpufortRT) # function
nodes.STDeallocate.register_backend("acc", dest_dialects, DeallocateHipGpufortRT)

backends.supported_destination_dialects.add("hipgpufort")
backends.register_postprocess_backend("acc", dest_dialects, Acc2HipGpufortRTPostprocess)
