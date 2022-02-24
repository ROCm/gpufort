# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import util
from gpufort import translator
from gpufort import indexer

from ... import opts

from .. import nodes
from .. import backends

from . import accbackends
from . import accnodes

## DIRECTIVES
# update
HIP_GPUFORT_RT_ACC_UPDATE = "{indent}call gpufort_acc_update_{kind}({var}{asyncr})\n"
# wait
HIP_GPUFORT_RT_ACC_WAIT = "{indent}call gpufort_acc_wait({queue}{asyncr})\n"
# init shutdown
HIP_GPUFORT_RT_ACC_INIT = "{indent}call gpufort_acc_init()\n"
HIP_GPUFORT_RT_ACC_SHUTDOWN = "{indent}call gpufort_acc_shutdown()\n"
# regions
HIP_GPUFORT_RT_ACC_ENTER_REGION = "{indent}call gpufort_acc_enter_region({region_kind})\n"
HIP_GPUFORT_RT_ACC_EXIT_REGION = "{indent}call gpufort_acc_exit_region({region_kind})\n"
## CLAUSES
# create
HIP_GPUFORT_RT_ACC_CREATE = "{indent}{dev_var} = gpufort_acc_create({var})\n"
HIP_GPUFORT_RT_ACC_NO_CREATE = "{indent}{dev_var} = gpufort_acc_no_create({var})\n"
HIP_GPUFORT_RT_ACC_COPYIN = "{indent}{dev_var} = gpufort_acc_copyin({var}{asyncr})\n"
HIP_GPUFORT_RT_ACC_COPY = "{indent}{dev_var} = gpufort_acc_copy({var}{asyncr})\n"
HIP_GPUFORT_RT_ACC_COPYOUT = "{indent}{dev_var} = gpufort_acc_copyout({var}{asyncr})\n"
HIP_GPUFORT_RT_ACC_PRESENT = "{indent}{dev_var} = gpufort_acc_present({var}{asyncr}{alloc})\n"

HIP_GPUFORT_RT_ACC_DELETE = "{indent}call gpufort_acc_delete({var}{alloc}{finalize})\n"

MAPPING_CLAUSE_2_TEMPLATE_MAP = {
    "create": HIP_GPUFORT_RT_ACC_CREATE,
    "no_create": HIP_GPUFORT_RT_ACC_NO_CREATE,
    "delete": HIP_GPUFORT_RT_ACC_DELETE,
    "copyin": HIP_GPUFORT_RT_ACC_COPYIN,
    "copy": HIP_GPUFORT_RT_ACC_COPY,
    "copyout": HIP_GPUFORT_RT_ACC_COPYOUT,
    "present": HIP_GPUFORT_RT_ACC_PRESENT
}

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
    last_decl_list_node.add_to_epilog(HIP_GPUFORT_RT_ACC_ENTER_REGION.format(\
        indent=indent,region_kind="implicit_region=.true."))
    for stendorreturn in stcontainer.return_or_end_statements():
        stendorreturn.add_to_prolog(HIP_GPUFORT_RT_ACC_EXIT_REGION.format(\
            indent=indent,region_kind="implicit_region=.true."))


class Acc2HipGpufortRT(accbackends.AccBackendBase):
    # clauses
    def _handle_async(self, queue=None, prefix=",asyncr="):
        """
        :return: Empty string if no queue was found
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
        """
        :return: If a finalize clause is present
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
        indent = self._stnode.first_line_indent()
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
                    result += template.format(indent=indent,var=var_expr,dev_var=deviceptr,\
                        asyncr=self._handle_async(),alloc="",finalize="")
                    temp_vars.add(deviceptr)
        return result, len(result), temp_vars

    def _handle_update(self):
        """Emits a acc_clause_update command for every variable in the list
        """
        result = ""
        indent = self._stnode.first_line_indent()
        # update host
        for parse_result in translator.tree.grammar.acc_mapping_clause.scanString(
                self._stnode.first_statement()):
            clause = parse_result[0][0]
            if clause.kind in ["self", "host", "device"]:
                clause_kind = "host" if clause.kind == "self" else clause.kind
                var_expressions = clause.var_expressions()
                for i, var_expr in enumerate(var_expressions):
                    result += HIP_GPUFORT_RT_ACC_UPDATE.format(
                        indent=indent,
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
        indent = self._stnode.first_line_indent()
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
            result += template.format(indent=indent,
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
        indent = stnode.first_line_indent()
        if stnode.is_init_directive():
            result += HIP_GPUFORT_RT_ACC_INIT.format(indent=indent)
        elif stnode.is_update_directive():
            partial_result, transformed = self._handle_update()
            if transformed:
                result += partial_result
        elif stnode.is_shutdown_directive():
            result += HIP_GPUFORT_RT_ACC_SHUTDOWN.format(indent=indent)
        else:
            ## Enter region commands must come first
            emit_enter_region = stnode.is_enter_data_directive()
            if emit_enter_region:
                region_kind = "unstructured=.true."
            else:
                region_kind = ""
            emit_enter_region = emit_enter_region or stnode.is_data_directive()
            emit_enter_region = emit_enter_region or stnode.is_parallel_directive(
            )
            emit_enter_region = emit_enter_region or stnode.is_parallel_loop_directive(
            )
            emit_enter_region = emit_enter_region or stnode.is_kernels_directive(
            )
            emit_enter_region = emit_enter_region or stnode.is_kernels_loop_directive(
            )
            if emit_enter_region:
                result += HIP_GPUFORT_RT_ACC_ENTER_REGION.format(
                    indent=indent, region_kind=region_kind)

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
            emit_exit_region = stnode.is_exit_data_directive()
            if emit_exit_region:
                region_kind = "unstructured=.true."
            else:
                region_kind = ""
            emit_exit_region = emit_exit_region or (
                stnode.is_end_directive() and
                stnode.find_substring("kernels") and
                not stnode.find_substring("loop"))
            emit_exit_region = emit_exit_region or (
                stnode.is_end_directive() and
                stnode.find_substring("parallel") and
                not stnode.find_substring("loop"))
            emit_exit_region = emit_exit_region or (
                stnode.is_end_directive() and stnode.find_substring("data"))
            if emit_exit_region:
                result += HIP_GPUFORT_RT_ACC_EXIT_REGION.format(
                    indent=indent, region_kind=region_kind)

        # _handle if
        condition = self._handle_if()
        if len(condition) and handle_if:
            result = "if ( {condition} ) then\n{result}\n endif".format(\
                    condition=condition, result=result.rstrip("\n"))

        # introduce the new variables
        stnode.parent.append_vars_to_decl_list(all_temp_vars)

        return result, len(result)


class AccLoopNest2HipGpufortRT(Acc2HipGpufortRT):

    def _handle_default(self):
        """
        Emits a acc_clause_present command for every variable in the list
        """
        stloopkernel = self._stnode
        ttloopkernel = stloopkernel.parse_result
        unmapped_arrays = ttloopkernel.all_unmapped_arrays()

        result = ""
        temp_vars = []
        "{indent}{dev_var} = gpufort_acc_present({var}{asyncr}{alloc})\n"
        template = HIP_GPUFORT_RT_ACC_PRESENT
        indent = stloopkernel.first_line_indent()
        default_clauses = translator.tree.grammar.acc_clause_default.searchString(
            stloopkernel.first_statement(), 1)
        if len(default_clauses):
            value = str(default_clauses[0][0][0]).lower()
            #print(value)
            if value == "present":
                for var_expr in unmapped_arrays:
                    deviceptr = dev_var_name(var_expr)
                    temp_vars.append(deviceptr)
                    result += template.format(indent=indent,
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
                        indent=indent,
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
        stloopkernel = self._stnode
        ttloopkernel = stloopkernel.parse_result
        arrays_in_body = ttloopkernel.arrays_in_body()
        indent = stloopkernel.first_line_indent()
        if stloopkernel.is_parallel_loop_directive(
        ) or stloopkernel.is_kernels_loop_directive():
            result, _ = Acc2HipGpufortRT.transform(
                self,
                joined_lines,
                joined_statements,
                statements_fully_cover_lines,
                index,
                handle_if=False)
        partial_result, transformed, temp_vars = self._handle_default()
        if transformed:
            stloopkernel.parent.append_vars_to_decl_list(temp_vars)
            result = "\n".join([result.rstrip("\n"), partial_result])
        partial_result, _ = nodes.STLoopNest.transform(
            stloopkernel, joined_lines, joined_statements,
            statements_fully_cover_lines, index)
        result = "\n".join([result.rstrip("\n"), partial_result])

        # handle default
        self._handle_default()
        # add wait call if necessary
        queue = self._handle_async(None, "")
        if not len(queue):
            result = result.rstrip(
                "\n") + "\n" + HIP_GPUFORT_RT_ACC_WAIT.format(
                    indent=indent, queue=queue, asyncr="")
        #if stloopkernel.is_parallel_loop_directive() or stloopkernel.is_kernels_loop_directive():
        if (stloopkernel.is_kernels_loop_directive() or
                stloopkernel.is_parallel_loop_directive()):
            result = result.rstrip(
                "\n") + "\n" + HIP_GPUFORT_RT_ACC_EXIT_REGION.format(
                    indent=indent, region_kind="")
        # wrap in ifdef if necessary
        condition = self._handle_if()
        if len(condition):
            result = "if ( {condition} ) then\n{result}\nelse\n {original}\n endif".format(\
                    condition=condition, result=result.rstrip("\n"), original="".join(stloopkernel._lines).rstrip("\n"))
        return result, len(result)


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
        ivar, _ = indexer.scope.search_index_for_var(index, parent_tag, var)
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
                indent=indent,var=host_var,asyncr="",\
                alloc=module_var+",or=gpufort_acc_event_"+map_kind,dev_var=dev_var))
            temp_vars.append(dev_var)
    if len(acc_present_calls):
        stcontainer.append_vars_to_decl_list(temp_vars)
        if implicit_region:
            add_implicit_region(stcontainer)
        for line in acc_present_calls:
            stallocate.add_to_epilog(line)
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
        ivar, _ = indexer.scope.search_index_for_var(index, parent_tag, var)
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
              indent=indent,var=host_var,asyncr="",finalize="",\
              alloc=module_var,dev_var=dev_var))
    for line in acc_delete_calls:
        stdeallocate.add_to_prolog(line)
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
                    cond = "{indent}if (associated({var})) "
                    acc_present_template = (
                        cond
                        + HIP_GPUFORT_RT_ACC_PRESENT.replace("{indent}", ""))
                else:
                    acc_present_template = HIP_GPUFORT_RT_ACC_PRESENT
                # find return and end, emit 1 new implicit region for all
                if ivar["declare_on_target"] in HIP_GPUFORT_RT_CLAUSES_OMP2ACC.keys(
                ):
                    map_kind = HIP_GPUFORT_RT_CLAUSES_OMP2ACC[
                        ivar["declare_on_target"]]
                    acc_present_calls += acc_present_template.format(\
                        indent=indent,var=host_var,asyncr="",\
                        alloc=module_var+",or=gpufort_acc_event_"+map_kind,dev_var=dev_var)
                    temp_vars.append(dev_var)
        if len(acc_present_calls):
            stcontainer.append_vars_to_decl_list(temp_vars)
            if implicit_region:
                add_implicit_region(stcontainer)
            last_decl_list_node.add_to_epilog(acc_present_calls)

dest_dialects = ["hipgpufort"]
accnodes.STAccDirective.register_backend(dest_dialects,Acc2HipGpufortRT()) # instance
accnodes.STAccLoopNest.register_backend(dest_dialects,AccLoopNest2HipGpufortRT())

nodes.STAllocate.register_backend("acc", dest_dialects, AllocateHipGpufortRT) # function
nodes.STDeallocate.register_backend("acc", dest_dialects, DeallocateHipGpufortRT)

backends.supported_destination_dialects.add("hipgpufort")
backends.register_postprocess_backend("acc", dest_dialects, Acc2HipGpufortRTPostprocess)