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
_ACC_INIT = "call gpufort_acc_init()\n"
_ACC_SHUTDOWN = "call gpufort_acc_shutdown()\n"
# regions
_ACC_ENTER_REGION = "call gpufort_acc_enter_region({region_kind})\n"
_ACC_EXIT_REGION = "call gpufort_acc_exit_region({region_kind})\n"

# clauses
_ACC_CREATE = "call gpufort_acc_ignore(gpufort_acc_create({var}))\n"
_ACC_NO_CREATE = "call gpufort_acc_ignore(gpufort_acc_no_create({var}))\n"
_ACC_PRESENT = "call gpufort_acc_ignore(gpufort_acc_present({var}{options}))\n"
_ACC_DELETE = "call gpufort_acc_delete({var}{options})\n"
_ACC_COPY = "call gpufort_acc_ignore(gpufort_acc_copy({var}{options}))\n"
_ACC_COPYIN = "call gpufort_acc_ignore(gpufort_acc_copyin({var}{options}))\n"
_ACC_COPYOUT = "call gpufort_acc_ignore(gpufort_acc_copyout({var}{options}))\n"
_ACC_PRESENT_OR_COPY = "call gpufort_acc_ignore(gpufort_acc_present_or_copy({var}{options}))\n"
_ACC_PRESENT_OR_COPYIN = "call gpufort_acc_ignore(gpufort_acc_present_or_copyin({var}{options}))\n"
_ACC_PRESENT_OR_COPYOUT = "call gpufort_acc_ignore(gpufort_acc_present_or_copyout({var}{options}))\n"

_DATA_CLAUSE_2_TEMPLATE_MAP = {
  "create": _ACC_CREATE,
  "no_create": _ACC_NO_CREATE,
  "delete": _ACC_DELETE,
  "copyin": _ACC_COPYIN,
  "copy": _ACC_COPY,
  "present_or_copyout": _ACC_PRESENT_OR_COPYOUT,
  "present_or_copyin": _ACC_PRESENT_OR_COPYIN,
  "present_or_copy": _ACC_PRESENT_OR_COPY,
  "present_or_copyout": _ACC_PRESENT_OR_COPYOUT,
  "present": _ACC_PRESENT
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

def _create_options_str(options,prefix=""):
    while(options.count("")):
      options.remove("")
    if len(options):
        return "".join([",",",".join(options)])
    else:
        return ""


class Acc2HipGpufortRT(accbackends.AccBackendBase):
    
    def _get_finalize_clause_expr(self):
        finalize_present = self.stnode.has_finalize_clause()
        finalize_expr = ""
        if finalize_present:
            finalize_expr = "finalize=True"
        return finalize_expr
    
    def _get_async_clause_expr(self):
        async_queue, async_present = self.stnode.get_async_queue()
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
            result.append(_ACC_WAIT.format(var="",options=""))
        elif wait_present and len(wait_queues):
            result.append(_ACC_WAIT.format(var=",".join(wait_queues),options=""))
        return result
 
    def _handle_data_clauses(self,async_expr,finalize_expr):
        result = [] 
        #
        for kind, args in self.stnode.get_matching_clauses(_DATA_CLAUSE_2_TEMPLATE_MAP.keys()):
            options = []
            if kind in _DATA_CLAUSES_WITH_ASYNC:
                options.append(async_expr)
            if kind in _DATA_CLAUSES_WITH_FINALIZE:
                options.append(finalize_expr)
            for var_expr in args:
                template = _DATA_CLAUSE_2_TEMPLATE_MAP[kind.lower()]
                options_str =_create_options_str(options)
                result.append(template.format(var=var_expr,options=options_str))
        return result

    def _handle_if_clause(self,result):
        condition, found_if = self.stnode.get_if_clause_condition()
        if found_if:
            result = [textwrap.dedent(l," "*2) for l in result]
            result.insert(0,"if ( {} ) then\n".format(condition))
            result.append("endif\n".format(condition))

    def _update_directive(self,async_expr):
        """Emits a acc_clause_update command for every variable in the list
        """
        #if self.stnode.is_directive(["acc","update"]):
        result = []
        options = [ async_expr ]
        for kind, args in self.stnode.get_matching_clauses(["self", "host", "device"]):
            for var_expr in args:
                result.append(_ACC_UPDATE.format(
                    var=var_expr,
                    options=_create_options_str(options),
                    kind=kind.lower()))
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
                  index=[],
                  handle_if=True):
        """
        :param line: An excerpt from a Fortran file, possibly multiple lines
        :type line: str
        :return: If the text was changed at all
        :rtype: bool
        """
        result = []

        if stnode.is_directive(["acc","init"]):
            result.append(_ACC_INIT)
        elif stnode.is_directive(["acc","shutdown"]):
            result.append(_ACC_SHUTDOWN)
        elif stnode.is_directive(["acc","update"]):
            result.append(self._handle_wait_clause())
            async_expr = self._get_async_clause_expr()
            result += self._update_directive(async_expr)
        elif stnode.is_directive(["acc","wait"]):
            result += self._wait_directive()
        else: # data regions
            ## Enter region commands must come first
            if (stnode.is_directive(["acc","enter","data"])
               or stnode.is_directive(["acc","data"])
               or stnode.is_directive(["acc","parallel"])
               or stnode.is_directive(["acc","parallel","loop"])
               or stnode.is_directive(["acc","kernels"])
               or stnode.is_directive(["acc","kernels","loop"])):
                if not stnode.is_directive["acc","data"]:
                    result += self._handle_wait_clause()  
                if stnode.is_directive(["acc","enter","data"]):
                    result.append(_ACC_ENTER_REGION.format(
                        options="unstructured=.true."))
                else:
                    result.append(_ACC_ENTER_REGION.format(
                        options=""))

            ## mapping clauses on data directives
            if (stnode.is_directive(["acc","enter","data"])
               or stnode.is_directive(["acc","exit","data"])
               or stnode.is_directive(["acc","data"])
               or stnode.is_directive(["acc","kernels"])):
                async_expr = self._get_async_clause_expr()
                finalize_expr = self._get_finalize_clause_expr();
                if len(finalize_expr) and not stnode.is_directive(["acc","exit","data"]):
                    raise util.error.SyntaxError("finalize clause may only appear on 'exit data' directive.")
                result += self._handle_data_clauses(async_expr,finalize_expr)

            ## Exit region commands must come last
            if (stnode.is_directive(["acc","exit","data"]) 
               or stnode.is_directive(["acc","end","data"])
               or stnode.is_directive(["acc","end","kernels"])
               or stnode.is_directive(["acc","end","parallel"])):
                if stnode.is_directive(["acc","exit","data"]):
                    result.append(_ACC_EXIT_REGION.format(
                        region_kind="unstructured=.true."))
                else:
                    result.append(_ACC_EXIT_REGION.format(
                        region_kind=""))
        # _handle if
        if handle_if:
            self._handle_if_clause(result)

        indent = stnode.first_line_indent()
        return textwrap.indent("".join(result),indent), len(result)

class AccLoopNest2HipGpufortRT(Acc2HipGpufortRT):

    def _mapping(clause_kind,var_expr,**kwargs):
        asyncr,_   = util.kwargs.get_value("asyncr","",**kwargs)
        finalize,_ = util.kwargs.get_value("finalize","",**kwargs)
        
        tokens = ["gpufort_acc_",clause_kind,"("]
        if clause_kind in _DATA_CLAUSE_2_TEMPLATE_MAP:
            tokens.append(var_expr)
            if len(asyncr) and clause.kind in _DATA_CLAUSES_WITH_ASYNC:
                tokens += [",",asyncr]
            if len(finalize) and clause.kind in _DATA_CLAUSES_WITH_FINALIZE:
                tokens += [",",finalize]
            tokens.append(")") 
            return "".join(tokens)
        else:
            raise util.parsing.SyntaxError("clause not supported") 
    
    def derive_kernel_call_arguments(self):
        """:return a list of arguments given the directives.
        """
        result = []
        mappings = kernel_args_to_acc_mappings_no_types(self.stnode.clauses,
                                                    self.stnode.kernels_args_tavars,
                                                    self.stnode.get_vars_present_per_default(),
                                                    AccLoopNest2HipGpufortRT._mapping,
                                                    finalize=self._get_finalize_clause_expr(),
                                                    asyncr=self._get_async_clause_expr())
        for var_expr, runtime_call in mappings:
            tokens = ["gpufort_array_wrap_device_ptr(",runtime_call,
                         "shape(",var_expr,"),","lbounds(",var_expr,"))"]
            result.append("".join(tokens))
        return result
        
    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[]):
        result = []
        stloopnest = self.stnode
        ttloopnest = stloopnest.parse_result
        #arrays_in_body = ttloopnest.arrays_in_body()
        if (stloopnest.is_directive(["acc","parallel","loop"])
           or stloopnest.is_directive(["acc","kernels","loop"])):
            result_directive, _ = Acc2HipGpufortRT.transform(
                self,
                joined_lines,
                joined_statements,
                statements_fully_cover_lines,
                index,
                handle_if=False)
            result.append(textwrap.dedent(result_directive).rstrip("\n"))
        
        queue, found_async = stloopnest.get_async_clause_queue()
        if not found_async:
            queue = "0"
        stloopkernel.stream_f_str = "gpufort_acc_get_stream({})".format(queue)
        stloopkernel.blocking_launch_f_str = ".{}.".format(str(not found_async))
       
        stloopnest.kernel_args_names = self.derive_kernel_call_arguments()
        result_loopnest, _ = nodes.STLoopNest.transform(
                                 stloopnest, joined_lines, joined_statements,
                                 statements_fully_cover_lines, index)
        result.append(textwrap.dedent(partial_result))
        
        if (stloopnest.is_directive(["acc","kernels","loop"]) or
           stloopnest.is_directive(["acc","parallel","loop"])):
            result.append(_ACC_EXIT_REGION.format(region_kind=""))
        
        self._handle_if_clause(result)

        indent = stloopnest.first_line_indent()
        return textwrap.indent("".join(result),indent), len(result)

@util.logging.log_entry_and_exit(opts.log_prefix)
def _add_implicit_region(stcontainer):
    last_decl_list_node = stcontainer.last_entry_in_decl_list()
    indent = last_decl_list_node.first_line_indent()
    last_decl_list_node.add_to_epilog(textwrap.indent(_ACC_ENTER_REGION,format(\
        region_kind="implicit_region=.true."),indent))
    for stendorreturn in stcontainer.return_or_end_statements():
        stendorreturn.add_to_prolog(textwrap.indent(_ACC_EXIT_REGION.format(\
            region_kind="implicit_region=.true."),indent))

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
    for var in stallocate.variable_names:
        ivar = indexer.scope.search_index_for_var(index, parent_tag, var)
        host_var = ivar["name"]
        is_local_var = host_var in local_var_names
        is_arg = host_var in dummy_arg_names
        is_used_module_var = not is_local_var and not is_arg
        is_allocatable_or_pointer = "allocatable" in ivar["qualifiers"] or\
                             "pointer" in ivar["qualifiers"]
        assert is_allocatable_or_pointer
        module_var = ",module_var=.true." if is_used_module_var else ""
        if not is_used_module_var:
            implicit_region = True
        if ivar["declare_on_target"] in _CLAUSES_OMP2ACC.keys():
            map_kind = _CLAUSES_OMP2ACC[
                ivar["declare_on_target"]]
            acc_present_calls.append(_ACC_PRESENT.format(\
                var=host_var,options="",\
                alloc=module_var+",or=gpufort_acc_event_"+map_kind,dev_var=dev_var))
    if len(acc_present_calls):
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
        is_local_var = host_var in local_var_names
        is_arg = host_var in dummy_arg_names
        is_used_module_var = not is_local_var and not is_arg
        is_allocatable_or_pointer = "allocatable" in ivar["qualifiers"] or\
                             "pointer" in ivar["qualifiers"]
        assert is_allocatable_or_pointer
        module_var = ",module_var=.true." if is_used_module_var else ""
        if ivar["declare_on_target"] in ["alloc", "to", "tofrom"]:
            acc_delete_calls.append(_ACC_DELETE.format(\
              var=host_var,options="",finalize="",\
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
        implicit_region = False
        for ivar in scope_vars:
            host_var = ivar["name"]
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
                        + _ACC_PRESENT.replace("", ""))
                else:
                    acc_present_template = _ACC_PRESENT
                # find return and end, emit 1 new implicit region for all
                if ivar["declare_on_target"] in _CLAUSES_OMP2ACC.keys(
                ):
                    map_kind = _CLAUSES_OMP2ACC[
                        ivar["declare_on_target"]]
                    acc_present_calls += acc_present_template.format(\
                        var=host_var,options="",\
                        alloc=module_var+",or=gpufort_acc_event_"+map_kind,dev_var=dev_var)
        if len(acc_present_calls):
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
