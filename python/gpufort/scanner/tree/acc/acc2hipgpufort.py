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

# update
_ACC_UPDATE = "call gpufort_acc_update_{kind}({var}{options})\n"
# wait
_ACC_WAIT = "call gpufort_acc_wait({queue}{options})\n"
# init shutdown
_ACC_INIT     = "call gpufort_acc_init()\n"
_ACC_SHUTDOWN = "call gpufort_acc_shutdown()\n"

# regions
_ACC_ENTER_REGION = "call gpufort_acc_enter_region({options})\n"
_ACC_EXIT_REGION = "call gpufort_acc_exit_region({options})\n"
_ACC_DATA_START      = "call gpufort_acc_data_start({mappings}{options})\n"
_ACC_DATA_END        = "call gpufort_acc_data_end({mappings}{options})\n"
_ACC_ENTER_EXIT_DATA = "call gpufort_acc_enter_exit_data({mappings}{options})\n"

# clauses
#_ACC_CREATE = "call gpufort_acc_ignore(gpufort_acc_create({var}))\n"
#_ACC_NO_CREATE = "call gpufort_acc_ignore(gpufort_acc_no_create({var}))\n"
#_ACC_PRESENT = "call gpufort_acc_ignore(gpufort_acc_present({var}{options}))\n"
#_ACC_DELETE = "call gpufort_acc_delete({var}{options})\n"
#_ACC_COPY = "call gpufort_acc_ignore(gpufort_acc_copy({var}{options}))\n"
#_ACC_COPYIN = "call gpufort_acc_ignore(gpufort_acc_copyin({var}{options}))\n"
#_ACC_COPYOUT = "call gpufort_acc_ignore(gpufort_acc_copyout({var}{options}))\n"
#_ACC_PRESENT_OR_CREATE = "call gpufort_acc_ignore(gpufort_acc_present_or_create({var}{options}))\n"
#_ACC_PRESENT_OR_COPYIN = "call gpufort_acc_ignore(gpufort_acc_present_or_copyin({var}{options}))\n"
#_ACC_PRESENT_OR_COPYOUT = "call gpufort_acc_ignore(gpufort_acc_present_or_copyout({var}{options}))\n"
#_ACC_PRESENT_OR_COPY = "call gpufort_acc_ignore(gpufort_acc_present_or_copy({var}{options}))\n"
_ACC_USE_DEVICE = "gpufort_acc_use_device({var},lbound({var}){options})\n"

_ACC_MAP_CREATE = "gpufort_map_acc_create({var})"
_ACC_MAP_NO_CREATE = "gpufort_map_acc_no_create({var})"
_ACC_MAP_PRESENT = "gpufort_map_acc_present({var})"
_ACC_MAP_DELETE = "call gpufort_map_acc_delete({var}"
_ACC_MAP_COPY = "gpufort_map_acc_copy({var})"
_ACC_MAP_COPYIN = "gpufort_map_acc_copyin({var})"
_ACC_MAP_COPYOUT = "gpufort_map_acc_copyout({var})"
_ACC_MAP_PRESENT_OR_CREATE = "gpufort_map_acc_present_or_create({var})"
_ACC_MAP_PRESENT_OR_COPYIN = "gpufort_map_acc_present_or_copyin({var})"
_ACC_MAP_PRESENT_OR_COPYOUT = "gpufort_map_acc_present_or_copyout({var})"
_ACC_MAP_PRESENT_OR_COPY = "gpufort_map_acc_present_or_copy({var})"

_ACC_MAP_DEC_STRUCT_REFS = "gpufort_map_dec_struct_refs({var})"

_DATA_CLAUSE_2_TEMPLATE_MAP = {
  "create": _ACC_MAP_CREATE,
  "no_create": _ACC_MAP_NO_CREATE,
  "delete": _ACC_MAP_DELETE,
  "copyin": _ACC_MAP_COPYIN,
  "copyout": _ACC_MAP_COPYOUT,
  "copy": _ACC_MAP_COPY,
  "present": _ACC_MAP_PRESENT,
  "present_or_create": _ACC_MAP_PRESENT_OR_CREATE,
  "present_or_copyin": _ACC_MAP_PRESENT_OR_COPYIN,
  "present_or_copyout": _ACC_MAP_PRESENT_OR_COPYOUT,
  "present_or_copy": _ACC_MAP_PRESENT_OR_COPY,
}
        
_DATA_CLAUSES_WITH_ASYNC = [
    "present",
    "copyin","copy","copyout",
    "present_or_copyin","present_or_copy","present_or_copyout",
  ]
_DATA_CLAUSES_WITH_FINALIZE = ["delete"] 

_CLAUSES_OMP2ACC = {
  "alloc": "create",
  "to": "copyin",
  "tofrom": "copy"
}

def _create_options_str(options):
    while(options.count("")):
      options.remove("")
    if len(options):
        return "".join([",",",".join(options)])
    else:
        return ""
        
def _create_mappings_str(mappings,indent):
    """Join and indent mappings."""
    return textwrap.indent(
        ",&\n".join(mappings),indent)

class Acc2HipGpufortRT(accbackends.AccBackendBase):
    
    def _get_finalize_clause_expr(self):
        finalize_present = self.stnode.has_finalize_clause()
        finalize_expr = ""
        if finalize_present:
            finalize_expr = "finalize=True"
        return finalize_expr
    
    def _get_async_clause_expr(self,staccdir):
        """
        :param STAccDirective staccdir: OpenACC directive scanner node
        """
        async_queue, async_present = staccdir.get_async_clause_queue()
        async_expr = ""
        if async_present and async_queue!=None:
            async_expr = "async={}".format(async_queue)
        elif async_present and async_queue==None:
            async_expr = "async=0"
        return async_expr

    def _handle_wait_clause(self):
        result = []
        wait_queues, wait_present = self.stnode.get_wait_clause_queues()
        wait_expr = ""
        if wait_present and not len(wait_queues):
            result.append(_ACC_WAIT.format(queue="",options=""))
        elif wait_present and len(wait_queues):
            result.append(_ACC_WAIT.format(queue="["+",".join(wait_queues)+"]",options=""))
        return result

    # TODO clean up
    def _handle_data_clauses(self,staccdir,index,
                             template=None):#,async_expr,finalize_expr):
        """
        Handle all data clauses of the current directive and/or of the preceding data clauses.
        :param str template: Overwrite the default mapping templates by a specific one.
                             In the context of structured data regions, can be used to enqueue
                             counter modification mappings for each original mapping.
        """
        result = [] 
        #
        for kind, args in staccdir.get_matching_clauses(_DATA_CLAUSE_2_TEMPLATE_MAP.keys()):
            for var_expr in args:
                if template == None:
                    template = _DATA_CLAUSE_2_TEMPLATE_MAP[kind.lower()]
                result.append(template.format(var=var_expr))
                if not opts.acc_map_derived_types: 
                    ivar = indexer.scope.search_index_for_var(index,staccdir.parent.tag(),var_expr)
                    #if ivar["f_type"] == "type":
                    #    result.pop(-1)
        return result

    def _handle_if_clause(self,staccdir,result):
        condition, found_if = staccdir.get_if_clause_condition()
        if found_if:
            result2 = [textwrap.indent(l," "*2) for l in result]
            result.clear()
            result.append("if ( {} ) then\n".format(condition))
            result += result2
            result.append("else\n".format(condition))
            original_snippet = textwrap.indent(textwrap.dedent("\n".join([l.rstrip("\n") for l in staccdir.lines()]))," "*2)
            result += original_snippet + "\n"
            result.append("endif\n".format(condition))
    
    def _update_directive(self,index,async_expr):
        """Emits a acc_clause_update command for every variable in the list
        """
        #if self.stnode.is_directive(["acc","update"]):
        result = []
        options = [ async_expr ]
        for kind, args in self.stnode.get_matching_clauses(["if","if_present"]):
            if kind == "if":
                options.append("=".join(["condition",args[0]]))
            elif kind == "if_present":
                options.append("if_present=.true.")
        for kind, args in self.stnode.get_matching_clauses(["self", "host", "device"]):
            for var_expr in args:
                result.append(_ACC_UPDATE.format(
                    var=var_expr,
                    options=_create_options_str(options),
                    kind=kind.lower()))
                if not opts.acc_map_derived_types: 
                    tag = indexer.scope.create_index_search_tag_for_var(var_expr)
                    ivar = indexer.scope.search_index_for_var(index,self.stnode.parent.tag(),tag)
                    if ("%" in tag and ivar["rank"] == 0
                       or ivar["f_type"] == "type"):
                            result.pop(-1)
        return result
    
    def _host_data_directive(self):
        """Emits an associate statement with a mapping
        of each host variable to the mapped device array.
        """
        mappings = []
        options = []
        for kind, args in self.stnode.get_matching_clauses(["if","if_present"]):
            if kind == "if":
                options.append("=".join(["condition",args[0]]))
            elif kind == "if_present":
                options.append("if_present=.true.")
        for kind, args in self.stnode.get_matching_clauses(["use_device"]):
            for var_expr in args:
                mappings.append(" => ".join([
                  var_expr,
                  _ACC_USE_DEVICE.format(var=var_expr,
                    options=_create_options_str(options)).rstrip(),
                  ]))
        result = []
        if len(mappings):
            result.append("associate (&\n")
            for mapping in mappings[:-1]:
                result.append("".join(["  ",mapping,",","&\n"]))
            result.append("".join(["  ",mappings[-1],"&\n"]))
            result.append(")\n")
        return result
    
    def _end_host_data_directive(self,async_expr):
        #if self.stnode.is_directive(["acc","update"]):
        result = [""]
        return result

    def _wait_directive(self):
        """Emits an acc_wait  command for every variable in the list
        """
        result = []
        queues = self.stnode.directive_args
        asyncr_list = []
        for kind, args in self.stnode.get_matching_clauses(["async"]):
            for var_expr in args:
                asyncr_list.append(var_expr)
        queue = ""
        asyncr = ""
        if len(queues):
            queue = "[{}]".format(",".join(queues))
        if len(asyncr_list):
            asyncr = ",[{}]".format(",".join(asyncr_list))
        result.append(_ACC_WAIT.format(queue=queue,
                                       options=asyncr))
        return result

    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[]):
        """
        :param line: An excerpt from a Fortran file, possibly multiple lines
        :type line: str
        :return: If the text was changed at all
        :rtype: bool
        """
        result = []

        stnode = self.stnode
        indent = stnode.first_line_indent()
        if stnode.is_directive(["acc","init"]):
            result.append(_ACC_INIT)
        elif stnode.is_directive(["acc","shutdown"]):
            result.append(_ACC_SHUTDOWN)
        elif stnode.is_directive(["acc","update"]):
            result += self._handle_wait_clause()
            async_expr = self._get_async_clause_expr()
            result += self._update_directive(index,async_expr)
        elif stnode.is_directive(["acc","wait"]):
            result += self._wait_directive()
        elif stnode.is_directive(["acc","host_data"]):
            result += self._host_data_directive()
        elif stnode.is_directive(["acc","end","host_data"]):
            result.append("end associate")
        elif (stnode.is_directive(["acc","parallel"])
             or stnode.is_directive(["acc","serial"])
             or stnode.is_directive(["acc","parallel","loop"])
             or stnode.is_directive(["acc","kernels","loop"])):
            assert False, "should not be called for serial, parallel (loop) and kernels loop directive"
        ## data and kernels regions
        elif stnode.is_directive(["acc","enter","data"]):
            result += self._handle_wait_clause()
            mappings = self._handle_data_clauses(stnode,index)
            options = [ self._get_async_clause_expr() ]
            
            result.append(_ACC_ENTER_EXIT_DATA.format(
                mappings="&\n"+_create_mappings_str(mappings,indent),
                options=_create_options_str(options)))
            # emit gpufort_acc_enter_exit_data
        elif stnode.is_directive(["acc","exit","data"]):
            result += self._handle_wait_clause()
            mappings = self._handle_data_clauses(stnode,index)
            options = [ self._get_async_clause_expr(),
                        self._get_finalize_clause_expr() ]
            result.append(_ACC_ENTER_EXIT_DATA.format(
                mappings="&\n"+_create_mappings_str(mappings,indent),
                options=_create_options_str(options)))
        elif stnode.is_directive(["acc","data"]):
            result += self._handle_wait_clause()
            mappings = self._handle_data_clauses(stnode,index)
            result.append(_ACC_DATA_START.format(
                mappings="&\n"+_create_mappings_str(mappings,indent),
                options=""))
        elif stnode.is_directive(["acc","end","data"]):
            stparentnode = stnode.parent_directive
            mappings = self._handle_data_clauses(
                stparentnode,index,template=_ACC_MAP_DEC_STRUCT_REFS)
            options = [ self._get_async_clause_expr(stparentnode) ]
            result.append(_ACC_DATA_END.format(
                mappings="&\n"+_create_mappings_str(mappings,indent),
                options=""))
        elif stnode.is_directive(["acc","kernels"]):
            result += self._handle_wait_clause()
            mappings = self._handle_data_clauses(stnode,index)
            options = [ self._get_async_clause_expr(stnode) ]
            
            result.append(_ACC_DATA_START.format(
                mappings="&\n"+_create_mappings_str(mappings,indent),
                options=_create_options_str(options)))
        elif stnode.is_directive(["acc","end","kernels"]):
            stparentnode = stnode.parent_directive
            mappings = self._handle_data_clauses(
                stparentnode,index,template=_ACC_MAP_DEC_STRUCT_REFS)
            options = [ self._get_async_clause_expr(stparentnode) ]
            
            result.append(_ACC_DATA_END.format(
                mappings="&\n"+_create_mappings_str(mappings,indent),
                options=_create_options_str(options)))
        # _handle if
        self._handle_if_clause(stnode,result)

        indent = stnode.first_line_indent()
        return textwrap.indent("".join(result),indent), len(result)

class AccComputeConstruct2HipGpufortRT(Acc2HipGpufortRT):

    def _map_array(clause_kind1,var_expr,tavar,**kwargs):
        asyncr,_   = util.kwargs.get_value("asyncr","",**kwargs)
        prepend_present,_ = util.kwargs.get_value("prepend_present",False,**kwargs)       

        if prepend_present and clause_kind1.startswith("copy"):
            clause_kind = "".join(["present_or_",clause_kind1])
        else:
            clause_kind = clause_kind1
        if clause_kind in _DATA_CLAUSE_2_TEMPLATE_MAP:
            runtime_call_tokens = ["gpufort_acc_",clause_kind,"("]
            runtime_call_tokens.append(var_expr)
            if len(asyncr) and clause_kind in _DATA_CLAUSES_WITH_ASYNC:
                runtime_call_tokens += [",",asyncr]
            runtime_call_tokens.append(")") 
            tokens = [
              "gpufort_array",str(tavar["c_rank"]),"_wrap_device_cptr(&\n",
              " "*4,"".join(runtime_call_tokens),
              ",shape(",var_expr,",kind=c_int),lbound(",var_expr,",kind=c_int))",
            ]
            return "".join(tokens)
        else:
            raise util.error.SyntaxError("clause not supported") 
    
    def derive_kernel_call_arguments(self):
        """
        :return a list of arguments given the directives.
        :note: Add all variables listed in the data region clauses as present
               Put them before this directive's own clauses.
               translator.analysis routine processes them in reversed order.
        """
        result = []
        
        acc_clauses = []
        data_and_kernels_ancestors  = self.stnode.get_acc_data_ancestors()
        kernels_ancestor = self.stnode.get_acc_kernels_ancestor()
        if kernels_ancestor != None:
            data_and_kernels_ancestors.append(kernels_ancestor)
        for staccdir in data_and_kernels_ancestors:
            for _,args in staccdir.get_matching_clauses(_DATA_CLAUSE_2_TEMPLATE_MAP):
                acc_clauses.append(("present",args))
        # TODO also consider acc declare'd variables here
       
        if not self.stnode.is_directive(["acc","kernels"]): # note: parent `acc kernels` directive lines are copied to each embedded compute construct
            acc_clauses += self.stnode.get_matching_clauses(_DATA_CLAUSE_2_TEMPLATE_MAP)
        
        kwargs = {
          "asyncr":self._get_async_clause_expr(self.stnode),
        }
        
        mappings = translator.analysis.kernel_args_to_acc_mappings_no_types(
                       acc_clauses,
                       self.stnode.kernel_args_tavars,
                       self.stnode.get_vars_present_per_default(),
                       AccComputeConstruct2HipGpufortRT._map_array,
                       **kwargs)
        for var_expr, argument in mappings:
            result.append(argument)
        return result
        
    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[]):
        result = []
        stloopnest = self.stnode
        ttloopnest = stloopnest.parse_result
        
        queue, found_async = stloopnest.get_async_clause_queue()
        if not found_async:
            queue = "0"
        stloopnest.stream_f_str = "gpufort_acc_get_stream({})".format(queue)
        stloopnest.async_launch_f_str = ".{}.".format(str(found_async)).lower()
       
        stloopnest.kernel_args_names = self.derive_kernel_call_arguments()
        result_loopnest, _ = nodes.STComputeConstruct.transform(
                                 stloopnest, joined_lines, joined_statements,
                                 statements_fully_cover_lines, index)
        result.append(textwrap.dedent(result_loopnest))
        
        self._handle_if_clause(stloopnest,result)

        indent = stloopnest.first_line_indent()
        return textwrap.indent("".join(result),indent), len(result)

@util.logging.log_entry_and_exit(opts.log_prefix)
def _add_structured_data_region(stcontainer,mappings):
    pass
    #stcontainer.append_to_decl_list([_ACC_DATA_START.format(
    #    mappings)])
    #stcontainer.prepend_to_return_or_end_statements([_ACC_DATA_END.format(
    #    mappings)])

def AllocateHipGpufortRT(stallocate, joined_statements, index):
    stcontainer = stallocate.parent
    parent_tag = stcontainer.tag()
    scope = indexer.scope.create_scope(index, parent_tag)
    scope_vars = scope["variables"]
    indent = stallocate.first_line_indent()
    local_var_names, dummy_arg_names = stcontainer.local_and_dummy_var_names(
        index)
    acc_present_calls = []
    implicit_region = False
    for var in [a[0] for a in stallocate.allocations]:
        ivar = indexer.scope.search_index_for_var(index, parent_tag, var)
        if opts.acc_map_derived_types or ivar["f_type"] != "type":
            var_expr = ivar["name"]
            is_local_var = var_expr in local_var_names
            is_arg = var_expr in dummy_arg_names
            is_used_module_var = not is_local_var and not is_arg
            is_allocatable_or_pointer = "allocatable" in ivar["attributes"] or\
                                 "pointer" in ivar["attributes"]
            assert is_allocatable_or_pointer # TODO emit error
            module_var = ",module_var=.true." if is_used_module_var else ""
            if not is_used_module_var:
                implicit_region = True
            declare = ivar["declare_on_target"]
            if declare in _CLAUSES_OMP2ACC.keys():
                map_kind = _CLAUSES_OMP2ACC[declare]
                acc_present_template =  _DATA_CLAUSE_2_TEMPLATE_MAP[map_kind]
                acc_present_calls.append(acc_present_template.format(
                    var=var_expr,options=module_var))
    if len(acc_present_calls):
        if implicit_region:
            _add_structured_data_region(stcontainer)
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
        if opts.acc_map_derived_types or ivar["f_type"] != "type":
            var_expr = ivar["name"]
            is_local_var = var_expr in local_var_names
            is_arg = var_expr in dummy_arg_names
            is_used_module_var = not is_local_var and not is_arg
            is_allocatable_or_pointer = "allocatable" in ivar["attributes"] or\
                                 "pointer" in ivar["attributes"]
            assert is_allocatable_or_pointer
            module_var = ",module_var=.true." if is_used_module_var else ""
            if ivar["declare_on_target"] in ["alloc", "to", "tofrom"]:
                acc_delete_calls.append(_ACC_DELETE.format(\
                  var=var_expr,options=module_var))
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
        acc_present_calls = []
        implicit_region = False
        for ivar in scope_vars:
            var_expr = ivar["name"]
            if opts.acc_map_derived_types or ivar["f_type"] != "type":
                is_local_var = var_expr in local_var_names
                is_arg = var_expr in dummy_arg_names
                is_used_module_var = not is_local_var and not is_arg
                is_allocatable = "allocatable" in ivar["attributes"]
                is_pointer = "pointer" in ivar["attributes"]
                if not is_allocatable:
                    if not is_used_module_var:
                        implicit_region = True
                    module_var = ",module_var=.true." if is_used_module_var else ""
                    # find return and end, emit 1 new implicit region for all
                    declare = ivar["declare_on_target"]
                    if declare in _CLAUSES_OMP2ACC.keys():
                        map_kind = "".join(["present_or_",_CLAUSES_OMP2ACC[declare]])
                        acc_present_template =  _DATA_CLAUSE_2_TEMPLATE_MAP[map_kind]
                        if is_pointer:
                            acc_present_template = "".join(["if (associated({var})) ",acc_present_template])
                        acc_present_calls.append(acc_present_template.format(
                            var=var_expr,options=module_var))
        if len(acc_present_calls):
            if implicit_region:
                _add_structured_data_region(stcontainer)
            last_decl_list_node.add_to_epilog(textwrap.indent("".join(acc_present_calls),indent))

dest_dialects = ["hipgpufort"]
accnodes.STAccDirective.register_backend(dest_dialects,Acc2HipGpufortRT()) # instance
accnodes.STAccComputeConstruct.register_backend(dest_dialects,AccComputeConstruct2HipGpufortRT())

nodes.STAllocate.register_backend("acc", dest_dialects, AllocateHipGpufortRT) # function
nodes.STDeallocate.register_backend("acc", dest_dialects, DeallocateHipGpufortRT)

backends.supported_destination_dialects.add("hipgpufort")
backends.register_postprocess_backend("acc", dest_dialects, Acc2HipGpufortRTPostprocess)
