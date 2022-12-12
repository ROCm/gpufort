# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
"""This package provides routines to convert 
OpenACC-related translator tree nodes into
data structures that per clause indicate
if the clause is specified and what
expressions where given as arguments.
All expressions are provided as translator trees.
"""
from gpufort import util

from .. import tree
from .. import optvals

from . import fortran

class AccConstructInfo:
    def __init__(self,
        device_type,
        is_serial,
        is_parallel,
        is_kernels):
        """
        Constructor.
        :param str device_type: Device type identifier such as `acc_device_nvidia`.
        :param bool are_serial: Expressions are obtained from a serial construct.
        :param bool are_parallel: Expressions are obtained from a parallel construct.
        :param bool are_kernels: Expressions are obtained from a kernels construct.
        Mappings and reductions are stored in dictionaries, where the key is the
        kind of mapping ('create','copyin',...) and the reduction operator, respectively,
        and the values are lists of rvalue expressions.
        """
        self.device_type = device_type
        self.is_serial = is_serial
        self.is_parallel = is_parallel 
        self.is_kernels = is_kernels
        self.num_gangs = optvals.OptionalSingleValue()
        self.num_workers = optvals.OptionalSingleValue()
        self.vector_length = optvals.OptionalSingleValue()
        self.default = optvals.OptionalSingleValue
        self.mappings = optvals.OptionalDictValue()
        self.private_vars = optvals.OptionalListValue()
        self.firstprivate_vars = optvals.OptionalListValue()
        self.reduction = optvals.OptionalDictValue()
        self.if_cond = optvals.OptionalSingleValue
        self.self_cond = optvals.OptionalSingleValue
        self.async_arg = optvals.OptionalSingleValue

    def walk_mapped_variables(self):
        """:return: Per mapped variable, a tuple consisting 
                    of the mapping_kind and the variable.
        """
        if self.mappings.specified:
            for (mapping_kind,var_list) in self.mappings:
                for var in var_list:
                      yield (mapping_kind,var)
    
    def walk_private_variables(self):
        """:return: Per private variable, a tuple consisting 
                    of the mapping_kind and the variable.
        """
        if self.private_vars.specified:
            for var in self.private_vars:
                yield ("private",var)
    
    def walk_firstprivate_variables(self):
        """:return: Per firstprivate variable, a tuple consisting 
                    of the mapping_kind and the variable.
        """
        if self.private_vars.specified:
            for var in self.private_vars:
                yield ("firstprivate",var)

class AccRoutineInfo:
    def __init__(self,device_type):
        """Constructor.
        :param str device_type: Device type identifier such as `acc_device_nvidia`."""
        self.device_type = device_type
        self.name =  optvals.OptionalSingleValue()
        self.gang = optvals.OptionalSingleValue()
        self.worker = optvals.OptionalSingleValue()
        self.vector = optvals.OptionalSingleValue()
        self.seq = optvals.OptionalFlag()
        self.bind = optvals.OptionalSingleValue()
        self.nohost = optvals.OptionalFlag()
        
class AccLoopInfo:
    def __init__(self,device_type):
        """Constructor.
        :param str device_type: Device type identifier such as `acc_device_nvidia`."""
        self.device_type = device_type
        self.gang = optvals.OptionalSingleValue()
        self.worker = optvals.OptionalSingleValue()
        self.vector = optvals.OptionalSingleValue()
        self.independent = optvals.OptionalFlag()
        self.auto = optvals.OptionalFlag()
        self.tile = optvals.OptionalListValue()
        self.collapse = optvals.OptionalSingleValue()
        self.private_vars = optvals.OptionalListValue()
        self.reduction = optvals.OptionalDictValue()

    def gang_parallelism():
        return ( self.gang.specified
               and not self.worker.specified
               and not self.vector.specified )
    def gang_parallelism():
        return ( self.gang.specified
               and not self.worker.specified
               and not self.vector.specified )
    def gang_parallelism():
        return ( self.gang.specified
               and not self.worker.specified
               and not self.vector.specified )
    
    def walk_private_variables(self):
        """:return: Per private variable, a tuple consisting 
                    of the mapping_kind and the variable.
        """
        if self.private_vars.specified:
            for var in self.private_vars:
                yield ("private",var)

class AccCombinedConstructInfo(AccConstructInfo,AccLoopInfo):
    def __init__(self,device_type,
                 is_parallel,
                 is_kernels):
        AccConstructInfo.__init__(self,
                                  device_type,
                                  False,
                                  is_parallel,
                                  is_kernels)
        AccLoopInfo.__init__(self,
                             device_type)

def analyze_directive(ttaccdirective,device_type):
    if isinstance(ttaccdirective,(tree.TTAccParallelLoop,
                                    tree.TTAccKernelsLoop)):
        result = AccCombinedConstructInfo(
                   device_type,
                   is_parallel = isinstance(ttaccdirective,tree.TTAccParallel),
                   is_kernels = isinstance(ttaccdirective,tree.TTAccKernels))
    elif isinstance(ttaccdirective,(tree.TTAccSerial,
                                    tree.TTAccParallel,
                                    tree.TTAccKernels)):
        result = AccConstructInfo(
                   device_type,
                   is_serial = isinstance(ttaccdirective,tree.TTAccSerial),
                   is_parallel = isinstance(ttaccdirective,tree.TTAccParallel),
                   is_kernels = isinstance(ttaccdirective,tree.TTAccKernels))
    elif isinstance(ttaccdirective,tree.TTAccLoop):
        result = AccLoopInfo(device_type)
    elif isinstance(ttaccdirective,tree.TTAccRoutine):
        result = AccRoutineInfo(device_type)
        if ttaccdirective.id != None:
            result.name.value = ttaccdirective.id
    else:
        assert False, ("only implemented for instances of "
                      + "tree.TTAccLoop"
                      + ", tree.TTAccRoutine"
                      + ", and tree.TTAccComputeConstructBase")
    
    for ttnode in ttaccdirective.walk_clauses_device_type(device_type):
        if isinstance(ttnode,tree.TTAccMappingClause):
            clause_kind = ttnode.kind
            var_list = ttnode.var_list
            if isinstance(ttnode,(tree.TTAccClausePrivate)):
                for var in var_list:
                    result.private_vars.value.append(var)
            elif isinstance(ttnode,(tree.TTAccClauseFirstprivate)):
                for var in var_list:
                    result.firstprivate_vars.value.append(var)
            else:
                result.mappings.specified = True
                if not clause_kind in result.mappings.value:
                    result.mappings[clause_kind] = []
                result.mappings.value[clause_kind] += var_list
        elif isinstance(ttnode,tree.TTAccClauseGang):
            result.gang.value = ttnode.arg
        elif isinstance(ttnode,tree.TTAccClauseWorker):
            result.worker.value = ttnode.arg
        elif isinstance(ttnode,tree.TTAccClauseVector):
            result.vector.value = ttnode.arg
        elif isinstance(ttnode,tree.TTAccClauseNohost):
            result.nohost.specified = True
        elif isinstance(ttnode,tree.TTAccClauseAuto):
            result.auto.specified = True
        elif isinstance(ttnode,tree.TTAccClauseIndependent):
            result.independent.specified = True
        elif isinstance(ttnode,tree.TTAccClauseNumGangs):
            result.num_gangs.value = ttnode.arg
        elif isinstance(ttnode,tree.TTAccClauseNumWorkers):
            result.num_workers.value = ttnode.arg
        elif isinstance(ttnode,tree.TTAccClauseVectorLength):
            result.vector_length.value = ttnode.arg
        elif isinstance(ttnode,tree.TTAccClauseCollapse):
            result.collapse.value = ttnode.arg
        elif isinstance(ttnode,tree.TTAccClauseTile):
            result.tile.value = ttnode.args 
        elif isinstance(ttnode,tree.TTAccClauseReduction):
            op = ttnode.op
            var_list = ttnode.var_list
            if op not in result.reduction:
                result.reduction.value[op] = []
            for var in var_list:
                result.reduction.value[op].append(var)
        elif isinstance(ttnode,tree.TTAccClauseDefault):
            result.default.value = ttnode.value
        elif isinstance(ttnode,tree.TTAccClauseIf):
            result.if_cond.value = ttnode.condition
        elif isinstance(ttnode,tree.TTAccClauseSelf):
            result.self_cond.value = ttnode.condition
        elif isinstance(ttnode,tree.TTAccClauseBind):
            result.bind.value = ttnode.ident
        elif isinstance(ttnode,tree.TTAccClauseWait):
            result.wait.value = ttnode.expressions
        elif isinstance(ttnode,tree.TTAccClauseAsync):
            result.async_arg.value = ttnode.expression
    return result

def find_rvalues_in_directive(ttaccdirective,rvalues):
    """Search through arguments of loop directive clauses and discover
    rvalues expressions.
    :param list rvalues: List, inout argument
    """
    lvalues = []
    for ttnode in ttaccdirective.walk_preorder():
        if isinstance(ttnode,(
            tree.TTAccClauseGang,
            tree.TTAccClauseTile,
            tree.TTAccClauseVector,
            tree.TTAccClauseWorker)):
              fortran.find_lvalues_and_rvalues(ttnode,lvalues,rvalues)
