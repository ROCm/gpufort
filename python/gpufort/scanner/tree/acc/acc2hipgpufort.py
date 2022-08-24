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
_ACC_INIT = "call gpufortrt_init()\n"
_ACC_SHUTDOWN = "call gpufortrt_shutdown()\n"

# regions
_ACC_DATA_START = "call gpufortrt_data_start({args})\n"
_ACC_DATA_END = "call gpufortrt_data_end({args})\n"
_ACC_ENTER_EXIT_DATA = "call gpufortrt_enter_exit_data({args})\n"

# clauses
#_ACC_USE_DEVICE = "gpufortrt_use_device({args},lbound({args}){options})\n"
_ACC_USE_DEVICE = "gpufortrt_use_device{rank}({args})\n"

_ACC_MAP_CREATE = "gpufortrt_map_create({args})"
_ACC_MAP_NO_CREATE = "gpufortrt_map_no_create({args})"
_ACC_MAP_PRESENT = "gpufortrt_map_present({args})"
_ACC_MAP_DELETE = "gpufortrt_map_delete({args})"
_ACC_MAP_COPY = "gpufortrt_map_copy({args})"
_ACC_MAP_COPYIN = "gpufortrt_map_copyin({args})"
_ACC_MAP_COPYOUT = "gpufortrt_map_copyout({args})"

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

def _make_map_args(var_expr,ivar,fixed_size_module_var=True):
    args=[var_expr]
    #if ivar["rank"] > 0:
    #    args.append("size({})".format(var_expr))
    return args

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
            async_expr = "async_arg={}".format(async_queue)
        elif async_present and async_queue==None:
            async_expr = "async_arg=gpufortrt_async_noval"
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
    def _handle_data_clauses(self,staccdir,index):
        """Handle all data clauses of the current directive and/or of the preceding data clauses."""
        result = [] 
        #
        for kind, var_exprs in staccdir.get_matching_clauses(_DATA_CLAUSE_2_TEMPLATE_MAP.keys()):
            for var_expr in var_exprs:
                template = _DATA_CLAUSE_2_TEMPLATE_MAP[kind.lower()]
                ivar = indexer.scope.search_index_for_var(index,staccdir.parent.tag(),var_expr)
                result.append(template.format(args=",".join(_make_map_args(var_expr,ivar))))
                # TODO hack, removes derived types from list
                if not opts.acc_map_derived_types: 
                    if ivar["f_type"] == "type":
                        result.pop(-1)
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
                options.append("if_arg="+options[0])
            elif kind == "if_present":
                options.append("if_present_arg=.true.")
        for kind, clause_args in self.stnode.get_matching_clauses(["self", "host", "device"]):
            for var_expr in clause_args:
                tag = indexer.scope.create_index_search_tag_for_var(var_expr)
                ivar = indexer.scope.search_index_for_var(index,self.stnode.parent.tag(),tag)
                result.append(_ACC_UPDATE.format(
                    args=_create_args_str(_make_map_args(var_expr,ivar)+options,indent="",sep=","),
                    kind=kind.lower().replace("host","self")))
                if not opts.acc_map_derived_types: 
                    if ("%" in tag and ivar["rank"] == 0
                       or ivar["f_type"] == "type"):
                            result.pop(-1)
        return result

    def _get_host_data_region_num(self):
        host_data_region_num = 0
        stcurrentdir = self.stnode
        while (isinstance(stcurrentdir,accnodes.STAccDirective)
               and stcurrentdir.is_directive(["acc","host_data"])):
            host_data_region_num +=1
            stcurrentdir = stcurrentdir.parent_directive
        return host_data_region_num

    def _host_data_directive(self,index):
        """Emits an associate statement with a mapping
        of each host variable to the mapped device array.
        """
        # TODO legacy, work with macros instead as a workaround
        # somehow get C pointer to `dimension(*)` variables
        # have use_device_pointer_arr and use_device_pointer variants
        # that can deal with type(*)[,dimension(*)] inputs and
        # simply obtain the c_loc of the argument
        indent = self.stnode.first_line_indent()
        result = [indent + "block\n"]
        options = []
        for kind, args in self.stnode.get_matching_clauses(["if","if_present"]):
            if kind == "if":
                options.append("if_arg="+args[0])
            elif kind == "if_present":
                options.append("if_present_arg=.true.")
        
        host_data_region_num = self._get_host_data_region_num()
        mapping_num = 1
        pointer_maps = []
        associations = []
        for kind, args in self.stnode.get_matching_clauses(["use_device"]):
            for var_expr in args:
                if not var_expr.isidentifier():
                    raise util.error.LimitationError("host_data use_device: argument must be plain identifiers, is: "+var_expr)
                ivar = indexer.scope.search_index_for_var(
                        index, self.stnode.parent.tag(), var_expr)
                if ivar["f_type"]=="type":
                    raise util.error.LimitationError("host_data use_device: argument '"+var_expr+"' must have basic datatype")
                rank = ivar["rank"]
                datatype = indexer.types.render_datatype(ivar) 
                pointer_name = "_".join(["gpufortrt_use_device_tmpvar",str(host_data_region_num),str(mapping_num)])
                pointer_decl = indent + datatype+",pointer :: " + pointer_name
                args = [pointer_name,var_expr]
                if rank > 0:
                    pointer_decl += "(" + ":,"*(rank-1) + ":" + ")\n"
                    if indexer.props.index_var_is_assumed_size_array(ivar):
                        if rank > 1:
                            colons=":,"*(rank-2)+":"
                            args.append(
                                "[shape({var}({colons},lbound({var},{rank}))),1]".format(
                                  var=var_expr,colons=colons,rank=rank))
                        else:
                            args.append("[1]")
                    else:
                        args.append("shape(" + var_expr + ")")
                    args.append("lbound(" + var_expr + ")")
                result.append(pointer_decl)
                pointer_maps.append(indent + "call " 
                              + _ACC_USE_DEVICE.format(args=_create_args_str(args+options,indent="",sep=","),
                                  rank=ivar["rank"]))
                associations.append(var_expr+" => "+pointer_name)
                mapping_num += 1
        result += pointer_maps
        result.append(indent + "associate (&\n")
        for assoc in associations[:-1]:
            result.append(indent + " " + assoc + ",&\n")
        result.append(indent + " " + associations[-1] + ")\n")
        return result
    
    def _end_host_data_directive(self):
        result = []
        #for kind, args in self.stnode.parent_directive.get_matching_clauses(["use_device"]):
        #    for var_expr in args:
        #        result.append("#undef " + var_expr + "\n")
        indent = self.stnode.first_line_indent()
        result.append(indent + "end associate\n")
        result.append(indent + "end block\n")
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
        if not opts.translate_other_directives:
            return (None, False)
        result = []
        stnode = self.stnode
        indent = stnode.first_line_indent()
        handle_if = True
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
            indent = ""
            handle_if = False
            result += self._host_data_directive(index)
        elif stnode.is_directive(["acc","end","host_data"]):
            indent = ""
            result += self._end_host_data_directive()
        elif stnode.is_directive(["acc","enter","data"]):
            result += self._handle_wait_clause()
            mappings = self._handle_data_clauses(stnode,index)
            options = [ self._get_async_clause_expr(stnode) ]
            result.append(_ACC_ENTER_EXIT_DATA.format(
                args=_create_mappings_str(mappings,options,indent)))
            # emit gpufortrt_enter_exit_data
        elif stnode.is_directive(["acc","exit","data"]):
            result += self._handle_wait_clause()
            mappings = self._handle_data_clauses(stnode,index)
            options = [ self._get_async_clause_expr(stnode),
                        self._get_finalize_clause_expr() ]
            result.append(_ACC_ENTER_EXIT_DATA.format(
                args=_create_mappings_str(mappings,options,indent)))
        elif stnode.is_directive(["acc","data"]):
            result += self._handle_wait_clause()
            mappings = self._handle_data_clauses(stnode,index)
            result.append(_ACC_DATA_START.format(
                args=_create_mappings_str(mappings,[],indent)))
        elif stnode.is_directive(["acc","end","data"]):
            stparentdir = stnode.parent_directive
            result.append(_ACC_DATA_END.format(
                args=_create_mappings_str([],[],indent)))
        elif stnode.is_directive(["acc","kernels"]):
            result += self._handle_wait_clause()
            mappings = self._handle_data_clauses(stnode,index)
            options = [ self._get_async_clause_expr(stnode) ]
            result.append(_ACC_DATA_START.format(
                args=_create_mappings_str(mappings,options,indent)))
        elif stnode.is_directive(["acc","end","kernels"]):
            stparentdir = stnode.parent_directive
            options = [ self._get_async_clause_expr(stparentdir) ]
            result.append(_ACC_DATA_END.format(
                args=_create_mappings_str([],options,indent)))
        if handle_if:
            self._handle_if_clause(stnode,result)
        return textwrap.indent("".join(result),indent), len(result)

class AccComputeConstruct2HipGpufortRT(Acc2HipGpufortRT):

    def _map_array1(clause_kind,var_expr,tavar,**kwargs):
        asyncr,_   = util.kwargs.get_value("asyncr","",**kwargs)

        if clause_kind in _DATA_CLAUSE_2_TEMPLATE_MAP:
            #runtime_call_tokens = ["gpufortrt_",clause_kind,"("]
            runtime_call_tokens = ["gpufortrt_deviceptr(", var_expr,")"]
            tokens = [
              "gpufort_array",str(tavar["c_rank"]),"_wrap_device_cptr(&\n",
              " "*4,"".join(runtime_call_tokens),
              ",shape(",var_expr,",kind=c_int),lbound(",var_expr,",kind=c_int))",
            ]
            return "".join(tokens)
        else:
            raise util.error.SyntaxError("clause not supported") 
   
    def _map_array2(clause_kind,var_expr,tavar,**kwargs):
        asyncr,_   = util.kwargs.get_value("asyncr","",**kwargs)

        if clause_kind in _DATA_CLAUSE_2_TEMPLATE_MAP:
            tokens = ["gpufortrt_map_",clause_kind,"(",var_expr]
            if len(asyncr) and clause_kind in _DATA_CLAUSES_WITH_ASYNC:
                tokens += [",",asyncr]
            tokens.append(")") 
            return "".join(tokens)
        else:
            raise util.error.SyntaxError("clause not supported") 
    
    def _map_array(*args,**kwargs):
        return (AccComputeConstruct2HipGpufortRT._map_array1(*args,**kwargs),
                AccComputeConstruct2HipGpufortRT._map_array2(*args,**kwargs))
    
    def derive_kernel_call_arguments(self):
        """
        :return a list of arguments given the directives.
        :note: Add all variables listed in the data region clauses as present
               Put them before this directive's own clauses.
               translator.analysis routine processes them in reversed order.
        """
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
        
        mapping_tuples = translator.analysis.kernel_args_to_acc_mappings_no_types(
                                 acc_clauses,
                                 self.stnode.kernel_args_tavars,
                                 self.stnode.get_vars_present_per_default(),
                                 AccComputeConstruct2HipGpufortRT._map_array,
                                 **kwargs)
        mappings  = []
        arguments = []
        for var_expr, inner in mapping_tuples:
            if isinstance(inner,tuple):
                arguments.append(inner[0])
                mappings.append(inner[1])
            else:
                arguments.append(inner)
                pass # scalars
        return mappings, arguments
        
    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[]):
        if not opts.translate_compute_constructs:
            return (None, False)
        result = []
        stcomputeconstruct = self.stnode
        indent = stcomputeconstruct.first_line_indent()
        ttcomputeconstruct = stcomputeconstruct.parse_result
        
        mappings, arguments = self.derive_kernel_call_arguments()
      
        # enter structured region region
        result += self._handle_wait_clause()
        options = [ self._get_async_clause_expr(stcomputeconstruct) ]
        result.append(_ACC_DATA_START.format(
            args=_create_mappings_str(mappings,options,indent)))

        # kernel launch 
        stcomputeconstruct.kernel_args_names = arguments
        result_computeconstruct, _ = nodes.STComputeConstruct.transform(
                                 stcomputeconstruct, joined_lines, joined_statements,
                                 statements_fully_cover_lines, index)
        result.append(textwrap.dedent(result_computeconstruct))

        # leave structured region
        options = [ self._get_async_clause_expr(stcomputeconstruct) ]
        result.append(_ACC_DATA_END.format(
            args=_create_mappings_str([],options,indent)))
        
        self._handle_if_clause(stcomputeconstruct,result)

        indent = stcomputeconstruct.first_line_indent()
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
            if not indexer.props.has_any_attribute(ivar,["allocatable","pointer"]):
                raise error.SyntaxError("variable '{}' that is subject to `allocate` intrinsic must have 'allocatable' or 'pointer' qualifier".
                        format(var_expr))
            if ivar["declare_on_target"] in ["alloc"]:
                enter_data_mappings.append(_ACC_MAP_CREATE.format(
                  args=_create_args_str(_make_map_args(var_expr,ivar),"",sep=",")))
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
            if not indexer.props.has_any_attribute(ivar,["allocatable","pointer"]):
                raise error.SyntaxError("variable '{}' that is subject to `deallocate` intrinsic must have 'allocatable' or 'pointer' qualifier".
                        format(var_expr))
            if ivar["declare_on_target"] in ["alloc"]:
                exit_data_mappings.append(_ACC_MAP_DELETE.format(
                  args=_create_args_str(_make_map_args(var_expr,ivar),"",sep=",")))
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
    stcontainer.prepend_to_procedure_exit_statements([data_end_str])

def Acc2HipGpufortRTPostprocess(stree, index):
    """:param stree: the full scanner tree
       :param staccdirectives: All acc directive tree accnodes."""
    accbackends.add_runtime_module_use_statements(stree,"gpufortrt_api")

    # TODO check if there is any acc used in the
    # construct at all, also check if there is a allocate/deallocate statement
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
            if (var_expr in local_var_names
               or var_expr in dummy_arg_names
               or indexer.props.index_var_is_module_var(ivar)):
                if opts.acc_map_derived_types or ivar["f_type"] != "type":
                    # find return and end/contains, emit 1 new implicit region for all
                    declare = ivar["declare_on_target"]
                    if declare in _CLAUSES_OMP2ACC.keys():
                        map_kind     = _CLAUSES_OMP2ACC[declare]
                        map_template = _DATA_CLAUSE_2_TEMPLATE_MAP[map_kind]
                        args=_make_map_args(var_expr,ivar)
                        if (indexer.props.index_var_is_module_var(ivar)
                           and not indexer.props.has_any_attribute(ivar,["allocatable","pointer"])):
                            args.append("never_deallocate=.true.") 
                        data_start_mappings.append(map_template.format(
                            args=_create_args_str(args,"",sep=",")))
        if len(data_start_mappings):
            _add_structured_data_region(stcontainer,data_start_mappings,data_end_mappings)

dest_dialects = ["hipgpufort"]
accnodes.STAccDirective.register_backend(dest_dialects,Acc2HipGpufortRT()) # instance
accnodes.STAccComputeConstruct.register_backend(dest_dialects,AccComputeConstruct2HipGpufortRT())

nodes.STAllocate.register_backend("acc", dest_dialects, AllocateHipGpufortRT) # function
nodes.STDeallocate.register_backend("acc", dest_dialects, DeallocateHipGpufortRT)

backends.supported_destination_dialects.add("hipgpufort")
backends.register_postprocess_backend("acc", dest_dialects, Acc2HipGpufortRTPostprocess)