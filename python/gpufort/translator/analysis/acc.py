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

class AccConstructInfo:
    def __init__(self,device_type,
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
        self.first_private_vars = optvals.OptionalListValue()
        self.reduction = optvals.OptionalDictValue()
        self.if_cond = optvals.OptionalSingleValue
        self.self_cond = optvals.OptionalSingleValue

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
    def check(self):
        num_specified = 0
        if self.gang.specified:
            num_specified += 1
        if self.worker.specified:
            num_specified += 1
        if self.vector.specified:
            num_specified += 1
        if self.seq.specified:
            num_specified += 1
        if num_specified == 0:
            raise util.error.SyntaxError("one of 'gang','worker','vector','seq' must be specified")
        elif num_specified > 1:
            raise util.error.SyntaxError("only one of 'gang','worker','vector','seq'")
        
class AccLoopInfo:
    def __init__(self,device_type):
        """Constructor.
        :param str device_type: Device type identifier such as `acc_device_nvidia`."""
        :param str device_type: Device type identifier such as `acc_device_nvidia`.
        self.device_type = device_type
        self.gang = optvals.OptionalSingleValue()
        self.worker = optvals.OptionalSingleValue()
        self.vector = optvals.OptionalSingleValue()
        self.independent = optvals.OptionalFlag()
        self.auto = optvals.OptionalFlag()
        self.tile = optvals.OptionalSingleValue()
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

class AccCombinedConstructInfo(AccConstructInfo,AccLoopInfo):
    def __init__(self,device_type,
                 is_parallel,
                 is_kernels):
        AccConstructInfo.__init__(device_type,
                                  is_parallel,
                                  is_kernels)
        AccLoopInfo.__init__(device_type)

class _TraverseDirectiveContext:
    def __init__(self):
        current_device_types = [] 
        encountered_input_device_type = False

def _analyse_directive_action(
      ttnode, # type: tree.Node
      parents, # type: list[Union[tree.TTNode,pyparsing.ParseResults,list]]
      result, # type: Union[AccConstructInfo,AccRoutineInfo,AccLoopInfo]
      ctx  # type: _TraverseDirectiveContext
    ):
    device_specific_clauses_apply_to_input_device_type = (
      not len(ctx.current_device_types) # no device_type clause encountered yet
      or result.device_type in ctx.current_device_types # 
      or (current_device_type == "*" and not ctx.encountered_device_type)
    )
    if isinstance(ttnode,TTAccMappingClause):
        clause_kind = ttnode.kind
        var_list = ttnode.var_list
        if node_kind not in ["private","firstprivate"]:
            result.mappings.specified = True
            if not node_kind in result.mappings.value:
                result.mappings[node_kind] = []
            result.mappings.value[node_kind] += var_list
        elif node_kind == "private":
            result.private_vars.value += var_list
        elif node_kind == "firstprivate":
            result.firstprivate_vars.value += var_list
    elif isinstance(ttnode,(tree.TTAccClauseGang)):
        if device_specific_clauses_apply_to_input_device_type:
            result.gang.value = ttnode.value
    elif isinstance(ttnode,(tree.TTAccClauseWorker)):
        if device_specific_clauses_apply_to_input_device_type:
            result.worker.value = ttnode.value
    elif isinstance(ttnode,(tree.TTAccClauseVector)):
        if device_specific_clauses_apply_to_input_device_type:
            result.vector.value = ttnode.value
    elif isinstance(ttnode,(tree.TTAccNoArgumentClause)):
        clause_kind = ttnode.kind.lower()
        if clause_kind == "nohost":
            result.nohost.specified = True
        if device_specific_clauses_apply_to_input_device_type:
            if clause_kind == "auto":
                result.auto.specified = True
            if clause_kind == "independent":
                result.independent.specified = True
    if isinstance(ttnode,(tree.TTAccClauseNumGangs)):
        if device_specific_clauses_apply_to_input_device_type:
            result.num_gangs.value = ttnode.value
    elif isinstance(ttnode,(tree.TTAccClauseNumWorkers)):
        if device_specific_clauses_apply_to_input_device_type:
            result.num_workers.value = ttnode.value
    elif isinstance(ttnode,(tree.TTAccClauseVectorLength)):
        if device_specific_clauses_apply_to_input_device_type:
            result.vector_length.value = ttnode.value
    elif isinstance(ttnode,(tree.TTAccClauseReduction)):
        op = ttnode.operator
        var_list = ttnode.vars
        if op not result.reduction:
            result.reduction.value[op] = []
        result.reduction.value[op] += var_list
    elif isinstance(ttnode,(tree.TTAccClauseDeviceType)):
        ctx.current_device_types = [d.fstr().lower() for d in ttnode.device_types]
        ctx.encountered_input_device_type = (
          result.device_type in ctx.current_device_types
        )
    elif isinstance(ttnode,(tree.TTAccClauseCollapse)):
        result.collapse.value = ttnode._value
    elif isinstance(ttnode,(tree.TTAccClauseTile)):
        result.tile.value = ttnode.tiles_per_dim 
    elif isinstance(ttnode,(tree.TTAccClauseDefault)):
        result.default.value = ttnode.value
    elif isinstance(ttnode,(tree.TTAccClauseIf)):
        result.if_cond.value = ttnode.condition
    elif isinstance(ttnode,(tree.TTAccClauseSelf)):
        result.self_cond.value = ttnode.condition
    elif isinstance(ttnode,(tree.TTAccClauseBind)):
        result.bind.value = ttnode.ident
    elif isinstance(ttnode,(tree.TTAccClauseWait)):
        result.wait.value = ttnode.expressions
    elif isinstance(ttnode,(tree.TTAccClauseAsync)):
        result.async.value = ttnode.expression

def analyze_directive(ttaccdirective,
                      device_type):
    if isinstance(ttaccdirective,tree.TTAccLoop):
        result = AccLoopInfo(device_type)
    elif isinstance(ttaccdirective,tree.TTAccRoutine):
        result = AccRoutineInfo(device_type)
        if ttaccdirective.id != None:
            result.name.value = ttaccdirective.id
    elif isinstance(ttaccdirective,(tree.TTAccSerial,
                                    tree.TTAccParallel,
                                    tree.TTAccKernels)):
        result = AccConstructInfo(
                   device_type,
                   is_serial = isinstance(ttaccdirective,TTAccSerial),
                   is_parallel = isinstance(ttaccdirective,TTAccParallel),
                   is_kernels = isinstance(ttaccdirective,TTAccKernels))
    elif isinstance(ttaccdirective,(tree.TTAccParallelLoop,
                                    tree.TTAccKernelsLoop)):
        result = AccCombinedConstructInfo(
                   device_type,
                   is_parallel = isinstance(ttaccdirective,TTAccParallel),
                   is_kernels = isinstance(ttaccdirective,TTAccKernels))
    else:
        assert False, ("only implemented for instances of "
                      + "tree.TTAccLoop"
                      + ", tree.TTAccRoutine"
                      + ", and tree.TTAccComputeConstructBase")
    tree.traversals.traverse(
        ttaccdirective,
        _visit_directive_action,
        tree.traversals.no_action,
        tree.traversals.no_crit,
        result,
        _TraverseDirectiveContext())
    return result

def _find_rvalues_in_directive_action(expr,parents,lvalues,rvalues)
  """Traversal action that searches through arguments of loop clauses and discover
  rvalues expressions."""
  if isinstance(expr,(
      tree.TTAccClauseGang,
      tree.TTAccClauseTile,
      tree.TTAccClauseVector,
      tree.TTAccClauseWorker)):
        _find_lvalues_and_rvalues(expr,lvalues,rvalues)

def find_rvalues_in_directive(ttaccdirective):
  """Search through arguments of loop directive clauses and discover
  rvalues expressions.
  :return: List of rvalues appearing in certain
           clauses of the directive.
  """
  lvalues, rvalues  = [], []
  tree.traversals.traverse(
      ttaccdirective,
      _find_rvalues_in_directive_action,
      tree.traversals.no_action,
      tree.traversals.no_crit,
      lvalues,
      rvalues)
  return rvalues
