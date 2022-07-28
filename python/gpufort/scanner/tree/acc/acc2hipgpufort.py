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
_ACC_UPDATE = "call gpufortrt_update_{kind}({args})\n"
# wait
#_ACC_WAIT = "call gpufortrt_wait({queue}{{args})\n"
_ACC_WAIT = "call gpufortrt_wait({args})\n"
# init shutdown
_ACC_INIT     = "call gpufortrt_init()\n"
_ACC_SHUTDOWN = "call gpufortrt_shutdown()\n"

# regions
_ACC_DATA_START      = "call gpufortrt_data_start({args})\n"
_ACC_DATA_END        = "call gpufortrt_data_end({args})\n"
_ACC_ENTER_EXIT_DATA = "call gpufortrt_enter_exit_data({args})\n"

# clauses
#_ACC_USE_DEVICE = "gpufortrt_use_device({args},lbound({args}){options})\n"
_ACC_USE_DEVICE = "gpufortrt_use_device({args})\n"

_ACC_MAP_CREATE = "gpufortrt_map_create({args})"
_ACC_MAP_NO_CREATE = "gpufortrt_map_no_create({args})"
_ACC_MAP_PRESENT = "gpufortrt_map_present({args})"
_ACC_MAP_DELETE = "call gpufortrt_map_delete({args})"
_ACC_MAP_COPY = "gpufortrt_map_copy({args})"
_ACC_MAP_COPYIN = "gpufortrt_map_copyin({args})"
_ACC_MAP_COPYOUT = "gpufortrt_map_copyout({args})"

_ACC_MAP_DEC_STRUCT_REFS = "gpufortrt_map_dec_struct_refs({args})"

_DATA_CLAUSE_2_TEMPLATE_MAP = {
  "present": _ACC_MAP_PRESENT,
  "create": _ACC_MAP_CREATE,
  "no_create": _ACC_MAP_NO_CREATE,
  "delete": _ACC_MAP_DELETE,
  "copyin": _ACC_MAP_COPYIN,
  "copyout": _ACC_MAP_COPYOUT,
  "copy": _ACC_MAP_COPY,
  "present_or_create": _ACC_MAP_CREATE,
  "present_or_copyin": _ACC_MAP_COPYIN,
  "present_or_copyout": _ACC_MAP_COPYOUT,
  "present_or_copy": _ACC_MAP_COPY,
  "pcreate": _ACC_MAP_CREATE,
  "pcopyin": _ACC_MAP_COPYIN,
  "pcopyout": _ACC_MAP_COPYOUT,
  "pcopy": _ACC_MAP_COPY,
}
        
_DATA_CLAUSES_WITH_ASYNC = [
  "present",
  "copyin","copy","copyout",
  "present_or_copyin","present_or_copy","present_or_copyout",
  "pcopyin","pcopy","pcopyout",
]

_CLAUSES_OMP2ACC = {
  "alloc": "create",
  "to": "copyin",
  "tofrom": "copy"
}
        
def _create_args_str(args,indent,sep=",&\n"):
    """Join and indent non-blank args"""
    while(args.count("")):
      args.remove("")
    if len(args):
        return textwrap.indent(sep.join(args),indent)
    return ""

def _create_mappings_str(mappings,options,indent):
    result = ""
    mappings_str = _create_args_str(mappings,indent) 
    options_str  = _create_args_str(options,indent)
    if len(mappings_str):
        result += "".join(["[&\n",mappings_str,"]"])
        if len(options_str):
            result += ",&\n"
    result += options_str
    return result

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
                result.append(template.format(args=var_expr))
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
        options   = [ async_expr ]
        for kind, options in self.stnode.get_matching_clauses(["if","if_present"]):
            if kind == "if":
                options.append("=".join(["condition",options[0]]))
            elif kind == "if_present":
                options.append("if_present=.true.")
        for kind, args in self.stnode.get_matching_clauses(["self", "host", "device"]):
            for var_expr in args:
                result.append(_ACC_UPDATE.format(
                    args=_create_args_str([var_expr]+options,indent="",sep=","),
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
        # TODO legacy, work with macros instead as a workaround
        # somehow get C pointer to `dimension(*)` variables
        # have use_device_pointer_arr and use_device_pointer variants
        # that can deal with type(*)[,dimension(*)] inputs and
        # simply obtain the c_loc of the argument
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
                  _ACC_USE_DEVICE.format(
                    args=_create_args_str([var_expr]+options,indent="",sep=","),
                    ).rstrip(),
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
            asyncr = "[{}]".format(",".join(asyncr_list))
        result.append(_ACC_WAIT.format(
            args=_create_args_str([queue,asyncr],indent="",sep=",")))
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
            async_expr = self._get_async_clause_expr(stnode)
            result += self._update_directive(index,async_expr)
        elif stnode.is_directive(["acc","wait"]):
            result += self._wait_directive()
        elif stnode.is_directive(["acc","host_data"]):
            result += self._host_data_directive()
        elif stnode.is_directive(["acc","end","host_data"]):
            # TODO legacy
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
                args="&\n"+_create_args_str(mappings+options,indent)))
            # emit gpufortrt_enter_exit_data
        elif stnode.is_directive(["acc","exit","data"]):
            result += self._handle_wait_clause()
            mappings = self._handle_data_clauses(stnode,index)
            options = [ self._get_async_clause_expr(),
                        self._get_finalize_clause_expr() ]
            result.append(_ACC_ENTER_EXIT_DATA.format(
                args="&\n"+_create_args_str(mappings+options,indent)))
        elif stnode.is_directive(["acc","data"]):
            result += self._handle_wait_clause()
            mappings = self._handle_data_clauses(stnode,index)
            result.append(_ACC_DATA_START.format(
                args=_create_mappings_str(mappings,[],indent)))
        elif stnode.is_directive(["acc","end","data"]):
            stparentdir = stnode.parent_directive
            mappings = self._handle_data_clauses(
                stparentdir,index,template=_ACC_MAP_DEC_STRUCT_REFS)
            result.append(_ACC_DATA_END.format(
                args=_create_mappings_str(mappings,[],indent)))
        elif stnode.is_directive(["acc","kernels"]):
            result += self._handle_wait_clause()
            mappings = self._handle_data_clauses(stnode,index)
            options = [ self._get_async_clause_expr(stnode) ]
            result.append(_ACC_DATA_START.format(
                args=_create_mappings_str(mappings,options,indent)))
        elif stnode.is_directive(["acc","end","kernels"]):
            stparentdir = stnode.parent_directive
            mappings = self._handle_data_clauses(
                stparentdir,index,template=_ACC_MAP_DEC_STRUCT_REFS)
            options = [ self._get_async_clause_expr(stparentdir) ]
            result.append(_ACC_DATA_END.format(
                args=_create_mappings_str(mappings,options,indent)))
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
            runtime_call_tokens = ["gpufortrt_",clause_kind,"("]
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
        stloopnest.stream_f_str = "gpufortrt_get_stream({})".format(queue)
        stloopnest.async_launch_f_str = ".{}.".format(str(found_async)).lower()
       
        stloopnest.kernel_args_names = self.derive_kernel_call_arguments()
        result_loopnest, _ = nodes.STComputeConstruct.transform(
                                 stloopnest, joined_lines, joined_statements,
                                 statements_fully_cover_lines, index)
        result.append(textwrap.dedent(result_loopnest))
        
        self._handle_if_clause(stloopnest,result)

        indent = stloopnest.first_line_indent()
        return textwrap.indent("".join(result),indent), len(result)

def AllocateHipGpufortRT(stallocate, joined_statements, index):
    stcontainer = stallocate.parent
    parent_tag = stcontainer.tag()
    scope = indexer.scope.create_scope(index, parent_tag)
    scope_vars = scope["variables"]
    indent = stallocate.first_line_indent()
    local_var_names, dummy_arg_names = stcontainer.local_and_dummy_var_names(
        index)
    enter_data_mappings = []
    for var in [a[0] for a in stallocate.allocations]:
        ivar = indexer.scope.search_index_for_var(index, parent_tag, var)
        if opts.acc_map_derived_types or ivar["f_type"] != "type":
            var_expr = ivar["name"]
            is_local_var = var_expr in local_var_names
            is_arg = var_expr in dummy_arg_names
            is_used_module_var = not is_local_var and not is_arg
            is_allocatable_or_pointer = "allocatable" in ivar["attributes"] or\
                                 "pointer" in ivar["attributes"]
            if not is_allocatable_or_pointer:
                raise error.SyntaxError("variable '{}' that is subject to `allocate` intrinsic must have 'allocatable' or 'pointer' qualifier".
                        format(var_expr))
            declare = ivar["declare_on_target"]
            if declare in _CLAUSES_OMP2ACC.keys():
                map_kind     = _CLAUSES_OMP2ACC[declare]
                map_template =  _DATA_CLAUSE_2_TEMPLATE_MAP[map_kind]
                enter_data_mappings.append(map_template.format(
                  args=_create_args_str([var_expr],"",sep=",")))
    if len(enter_data_mappings):
        enter_data_str = _ACC_ENTER_EXIT_DATA.format(
            args=_create_mappings_str(enter_data_mappings,[]," "*2))
        stallocate.add_to_epilog(textwrap.indent(enter_data_str,indent))
    return joined_statements, False


def DeallocateHipGpufortRT(stdeallocate, joined_statements, index):
    stcontainer = stdeallocate.parent
    parent_tag = stcontainer.tag()
    scope = indexer.scope.create_scope(index, parent_tag)
    scope_vars = scope["variables"]
    indent = stdeallocate.first_line_indent()
    local_var_names, dummy_arg_names = stcontainer.local_and_dummy_var_names(
        index)
    exit_data_mappings = []
    for var in stdeallocate.variable_names:
        ivar = indexer.scope.search_index_for_var(index, parent_tag, var)
        if opts.acc_map_derived_types or ivar["f_type"] != "type":
            var_expr = ivar["name"]
            is_local_var = var_expr in local_var_names
            is_arg = var_expr in dummy_arg_names
            is_used_module_var = not is_local_var and not is_arg
            is_allocatable_or_pointer = "allocatable" in ivar["attributes"] or\
                                 "pointer" in ivar["attributes"]
            if not is_allocatable_or_pointer:
                raise error.SyntaxError("variable '{}' that is subject to `deallocate` intrinsic must have 'allocatable' or 'pointer' qualifier".
                        format(var_expr))
            if ivar["declare_on_target"] in ["alloc", "to", "tofrom"]:
                exit_data_mappings.append(_ACC_MAP_DELETE.format(
                  args=_create_args_str([var_expr],"",sep=",")))
    if len(exit_data_mappings):
        exit_data_str = _ACC_ENTER_EXIT_DATA.format(
            args=_create_mappings_str(exit_data_mappings,[]," "*2))
        stdeallocate.add_to_prolog(textwrap.indent(exit_data_str,indent))
    return joined_statements, False

@util.logging.log_entry_and_exit(opts.log_prefix)
def _add_structured_data_region(stcontainer,data_start_mappings,data_end_mappings):
    data_start_str = _ACC_DATA_START.format(
        args=_create_mappings_str(data_start_mappings,[]," "*2))
    data_end_str = _ACC_DATA_END.format(
        args=_create_mappings_str(data_end_mappings,[]," "*2))
    stcontainer.append_to_decl_list([data_start_str])
    stcontainer.prepend_to_return_or_end_statements([data_end_str])

def Acc2HipGpufortRTPostprocess(stree, index):
    """:param stree: the full scanner tree
       :param staccdirectives: All acc directive tree accnodes."""
    accbackends.add_runtime_module_use_statements(stree,"gpufortrt")

    # TODO check if there is any acc used in the
    # construct at all
    # TODO handle directly via directives; only variables occuring
    # in directives need to be available on device
    containers = stree.find_all(\
      lambda node: type(node) in [nodes.STProgram,nodes.STProcedure],
      recursively=True)
    for stcontainer in containers:
        scope = indexer.scope.create_scope(index, stcontainer.tag())
        scope_vars = scope["variables"]
        local_var_names, dummy_arg_names = stcontainer.local_and_dummy_var_names(
            index)
        # TODO also process type members
        data_start_mappings = []
        data_end_mappings   = []
        for ivar in scope_vars:
            var_expr = ivar["name"]
            if opts.acc_map_derived_types or ivar["f_type"] != "type":
                is_local_var = var_expr in local_var_names
                is_arg = var_expr in dummy_arg_names
                is_used_module_var = not is_local_var and not is_arg
                is_allocatable = "allocatable" in ivar["attributes"]
                is_pointer = "pointer" in ivar["attributes"]
                if not is_allocatable:
                    module_var = "module_var=.true." if is_used_module_var else ""
                    # find return and end, emit 1 new implicit region for all
                    declare = ivar["declare_on_target"]
                    if declare in _CLAUSES_OMP2ACC.keys():
                        map_kind     = _CLAUSES_OMP2ACC[declare]
                        map_template = _DATA_CLAUSE_2_TEMPLATE_MAP[map_kind]
                        # TODO legacy
                        #if is_pointer:
                        #    map_template = "".join(["if (associated({var})) ",map_template])
                        data_start_mappings.append(map_template.format(
                            args=_create_args_str([var_expr,module_var],"",sep=",")))
                        data_end_mappings.append(_ACC_MAP_DEC_STRUCT_REFS.format(
                            args=_create_args_str([var_expr],"",sep=",")))
        if len(data_start_mappings):
            _add_structured_data_region(stcontainer,data_start_mappings,data_end_mappings)

dest_dialects = ["hipgpufort"]
accnodes.STAccDirective.register_backend(dest_dialects,Acc2HipGpufortRT()) # instance
accnodes.STAccComputeConstruct.register_backend(dest_dialects,AccComputeConstruct2HipGpufortRT())

nodes.STAllocate.register_backend("acc", dest_dialects, AllocateHipGpufortRT) # function
nodes.STDeallocate.register_backend("acc", dest_dialects, DeallocateHipGpufortRT)

backends.supported_destination_dialects.add("hipgpufort")
backends.register_postprocess_backend("acc", dest_dialects, Acc2HipGpufortRTPostprocess)
