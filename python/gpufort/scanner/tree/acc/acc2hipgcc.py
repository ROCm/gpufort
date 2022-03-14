# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import translator
from gpufort import util

from ... import opts

from .. import nodes
from .. import backends

from . import accbackends
from . import accnodes

# init shutdown
HIP_GCC_RT_ACC_INIT = "acc_init({devicetype})"
HIP_GCC_RT_ACC_SHUTDOWN = "acc_shutdown({devicetype})"

# update host
HIP_GCC_RT_ACC_UPDATE_SELF = "acc_update_self({var})"
HIP_GCC_RT_ACC_UPDATE_SELF_ASYNC = "acc_update_self_async({var}{asyncr})"
HIP_GCC_RT_ACC_UPDATE_DEVICE = "acc_update_device({var})"
HIP_GCC_RT_ACC_UPDATE_DEVICE_ASYNC = "acc_update_device_async({var}{asyncr})"
HIP_GCC_RT_ACC_UPDATE_MAP = { # [self][asyncr]
    False: {
        False: HIP_GCC_RT_ACC_UPDATE_DEVICE,
        True: HIP_GCC_RT_ACC_UPDATE_DEVICE_ASYNC
    },
    True: {
        False: HIP_GCC_RT_ACC_UPDATE_SELF,
        True: HIP_GCC_RT_ACC_UPDATE_SELF_ASYNC
    }
}
# wait
HIP_GCC_RT_ACC_WAIT_ALL = "acc_wait_all()"
HIP_GCC_RT_ACC_WAIT_ALL_ASYNC = "acc_wait_all_async({asyncr})"
HIP_GCC_RT_ACC_WAIT = "acc_wait({arg})"
HIP_GCC_RT_ACC_WAIT_ASYNC = "acc_wait_async({arg}{asyncr})"
#HIP_GCC_RT_ACC_WAIT_DEVICE       = "acc_wait_device({arg}{device})"
#HIP_GCC_RT_ACC_WAIT_DEVICE_ASYNC = "acc_wait_device({arg}{asyncr}{device})"
HIP_GCC_RT_ACC_WAIT_MAP = { # [arg][asyncr]
    False: {
        False: HIP_GCC_RT_ACC_WAIT_ALL,
        True: HIP_GCC_RT_ACC_WAIT_ALL_ASYNC
    },
    True: {
        False: HIP_GCC_RT_ACC_WAIT,
        True: HIP_GCC_RT_ACC_WAIT_ASYNC
    }
}
# delete
HIP_GCC_RT_ACC_DELETE = "acc_delete({var})"
HIP_GCC_RT_ACC_DELETE_ASYNC = "acc_delete_async({var}{asyncr})"
HIP_GCC_RT_ACC_DELETE_FINALIZE = "acc_delete_finalize({var})"
HIP_GCC_RT_ACC_DELETE_FINALIZE_ASYNC = "acc_delete_finalize_async({var}{asyncr})"
HIP_GCC_RT_ACC_DELETE_MAP = { # [finalize][asyncr]
    False: {
        False: HIP_GCC_RT_ACC_DELETE,
        True: HIP_GCC_RT_ACC_DELETE_ASYNC
    },
    True: {
        False: HIP_GCC_RT_ACC_DELETE_FINALIZE,
        True: HIP_GCC_RT_ACC_DELETE_FINALIZE_ASYNC
    }
}
# acc_deviceptr
HIP_GCC_RT_ACC_DEVICEPTR = "acc_deviceptr({var})" # usually not available in openbackends.f90
# GCC LIBGOMP specific internal routines for /acc enter/exit data and acc data
HIP_GCC_RT_GOACC_ENTER_EXIT_DATA = "goacc_enter_exit_data({device}{mappings}{asyncr}{wait})"
HIP_GCC_RT_GOACC_DATA_START = "goacc_data_start({device}{mappings}{asyncr})"
HIP_GCC_RT_GOACC_DATA_END = "goacc_data_end()"
# GCC LIBGOMP specific helper functions
HIP_GCC_RT_MAP_CREATE = "goacc_map_create({var})"
HIP_GCC_RT_MAP_NO_CREATE = "goacc_map_no_create({var})"
HIP_GCC_RT_MAP_DELETE = "goacc_map_delete({var})"
HIP_GCC_RT_MAP_COPYIN = "goacc_map_copyin({var})"
HIP_GCC_RT_MAP_COPY = "goacc_map_copy({var})"
HIP_GCC_RT_MAP_COPYOUT = "goacc_map_copyout({var})"
HIP_GCC_RT_MAP_PRESENT = "goacc_map_present({var})"


class Acc2HipGccRT(accbackends.AccBackendBase):

    def _create_mappings(self, parse_result, prefix=",mappings="):
        mappings = []
        for clause in translator.tree.find_all(parse_result,
                                          translator.tree.TTAccMappingClause):
            if clause.kind == "present":
                mappings += [
                    HIP_GCC_RT_MAP_PRESENT.format(var=expr)
                    for expr in clause.var_expressions()
                ]
            elif clause.kind == "create":
                mappings += [
                    HIP_GCC_RT_MAP_CREATE.format(var=expr)
                    for expr in clause.var_expressions()
                ]
            elif clause.kind == "no_create":
                mappings += [
                    HIP_GCC_RT_MAP_NO_CREATE.format(var=expr)
                    for expr in clause.var_expressions()
                ]
            elif clause.kind == "delete":
                mappings += [
                    HIP_GCC_RT_MAP_DELETE.format(var=expr)
                    for expr in clause.var_expressions()
                ]
            elif clause.kind == "copy":
                mappings += [
                    HIP_GCC_RT_MAP_COPY.format(var=expr)
                    for expr in clause.var_expressions()
                ]
            elif clause.kind == "copyin":
                mappings += [
                    HIP_GCC_RT_MAP_COPYIN.format(var=expr)
                    for expr in clause.var_expressions()
                ]
            elif clause.kind == "copyout":
                mappings += [
                    HIP_GCC_RT_MAP_COPYOUT.format(var=expr)
                    for expr in clause.var_expressions()
                ]
        if len(mappings):
            return prefix + "[" + ",".join(mappings) + "]"
        else:
            return ""

    def _handle_async(self, parse_result, prefix=",asyncr="):
        clause = translator.find_first(parse_result,
                                       translator.tree.TTAccClauseAsync)
        if clause != None:
            value = clause.expression()
            if str(value) == str(CLAUSE_VALUE_NOT_SPECIFIED):
                return prefix + "acc_async_noval"
            else:
                return prefix + value
        else:
            return ""

    def _handle_wait(self,
                     parse_result,
                     prefix=",wait=",
                     wrap_in_brackets=True):
        clause = translator.find_first(parse_result,
                                       translator.tree.TTAccClauseWait)
        if clause != None:
            result = ",".join(clause.expressions())
            if wrap_in_brackets:
                result = "[" + result + "]"
            return prefix + result
        else:
            return ""

    def _handle_device(self, parse_result):
        clause = translator.find_first(parse_result,
                                       translator.tree.TTAccClauseDevice)
        if clause != None:
            result = ",".join(clause.expressions())
            if wrap_in_brackets:
                result = "[" + result + "]"
            return prefix + result
        else:
            return ""

    def _handle_if(self, parse_result):
        clause = translator.find_first(parse_result,
                                       translator.tree.TTAccClauseIf)
        if clause != None:
            return clause.condition()
        else:
            return ""

    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[],
                  handle_if=False):
        f_snippet = joined_statements
        result = ""
        try:
            parse_result = translator.tree.grammar.acc_host_directive.parseString(
                f_snippet)[0]
            #
            if type(parse_result) in [translator.tree.TTAccData,\
                    translator.tree.TTAccParallel,translator.tree.TTAccParallelLoop,
                    translator.tree.TTAccKernels,translator.tree.TTAccKernelsLoop]:
                result = HIP_GCC_RT_GOACC_DATA_START.format(\
                  device   = "acc_device_default",\
                  mappings = self._create_mappings(parse_result),\
                  asyncr    = self._handle_async(parse_result))
            elif type(parse_result) in [
                    translator.tree.TTAccEnterData,
                    translator.tree.TTAccExitData
            ]:
                result = HIP_GCC_RT_GOACC_ENTER_EXIT_DATA.format(\
                  device   = "acc_device_default",\
                  mappings = self._create_mappings(parse_result),\
                  asyncr    = self._handle_async(parse_result),\
                  wait     = self._handle_wait(parse_result))
            elif type(parse_result) is translator.tree.TTAccEndData:
                result = HIP_GCC_RT_GOACC_DATA_END
            elif type(parse_result) is translator.tree.TTAccWait:
                arg = self._handle_wait(parse_result, "", False)
                asyncr = self._handle_async(parse_result)
                result = HIP_GCC_RT_ACC_WAIT_MAP[len(arg)][len(asyncr)].format(\
                  arg=arg,asyncr=asyncr)
                pass
            elif type(parse_result) is translator.tree.TTAccUpdate:
                host = self._handle_self(parse_result)
                device = self._handle_device(parse_result)
                asyncr = self._handle_async(parse_result, "")
                result = HIP_GCC_RT_ACC_UPDATE_MAP[len(host)][len(asyncr)].format(\
                  arg=arg,asyncr=asyncr)
            elif type(parse_result) is translator.tree.TTAccRoutine:
                pass
            elif type(parse_result) is translator.tree.TTAccDeclare:
                pass
            else:
                print("failed: " + f_snippet) #TODO
                return f_snippet, False
            return "call " + result + "\n", True
        except Exception as e:
            print("failed: " + f_snippet) #TODO
            return f_snippet, False


class AccLoopNest2HipGccRT(Acc2HipGccRT):

    def transform(self,
                  joined_lines,
                  joined_statements,
                  statements_fully_cover_lines,
                  index=[]):
        stnode = self.stnode
        indent = stnode.first_line_indent()
        result = ""
        if (stnode.is_directive(["acc","parallel","loop"])
           or stnode.is_directive(["acc","kernels","loop"])):
            partial_result, _ = Acc2HipGccRT.transform(
                self,
                joined_lines,
                joined_statements,
                statements_fully_cover_lines,
                index,
                handle_if=False)
            result = partial_result
        partial_result, _ = nodes.STLoopNest.transform(
            stnode, joined_lines, joined_statements,
            statements_fully_cover_lines, index)
        result += partial_result
        # add wait call if necessary
        arg = self._handle_async(None, "")
        if not len(arg):
            result += "\n" + indent + "call " + HIP_GCC_RT_ACC_WAIT_ALL.format(
                arg=arg, asyncr="") + "\n"
        if (stnode.is_directive(["acc","parallel","loop"])
           or stnode.is_directive(["acc","kernels","loop"])):
            result += indent + "call " + HIP_GCC_RT_GOACC_DATA_END + "\n"

        # wrap in if-then-else-endif if necessary
        f_snippet = joined_statements
        parse_result = translator.tree.grammar.acc_host_directive.parseString(
            f_snippet)[0]
        condition = self._handle_if(parse_result)
        if len(condition):
            result = "if ( {condition} ) then\n{result}\nelse\n {original}\n endif".format(\
              condition=condition,result=result.rstrip("\n"),original="".join(stnode._lines).rstrip("\n"))
        return result, len(result)


def AllocateHipGccRT(stallocate, index):
    return ""


def DeallocateHipGccRT(stdeallocate, index):
    return ""


def Acc2HipGccRTPostprocess(stree, index):
    accbackends.add_runtime_module_use_statements(stree,"openacc_gomp")

dest_dialects = ["hipgcc"]
accnodes.STAccDirective.register_backend(dest_dialects,Acc2HipGccRT()) # instance
accnodes.STAccLoopNest.register_backend(dest_dialects,AccLoopNest2HipGccRT())

nodes.STAllocate.register_backend("acc", dest_dialects, AllocateHipGccRT) # function
nodes.STDeallocate.register_backend("acc", dest_dialects, DeallocateHipGccRT)

backends.supported_destination_dialects.add("hipgcc")
backends.register_postprocess_backend("acc", dest_dialects, Acc2HipGccRTPostprocess)