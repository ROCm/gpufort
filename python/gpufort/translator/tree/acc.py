# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import util

from .. import opts
from .. import conv

from . import base
from . import traversals
from . import directives

class TTAccClause(base.TTNode):
    """Base class."""   
    pass

class TTAccClauseSeq(TTAccClause):
    kind = "seq" # will be available as class and instance var, separately
class TTAccClauseAuto(TTAccClause):
    kind = "auto"
class TTAccClauseIndependent(TTAccClause):
    kind = "independent"
class TTAccClauseRead(TTAccClause):
    kind = "read"
class TTAccClauseWrite(TTAccClause):
    kind = "write"
class TTAccClauseCapture(TTAccClause):
    kind = "capture"
class TTAccClauseUpdate(TTAccClause):
    kind = "update"
class TTAccClauseNohost(TTAccClause):
    kind = "nohost"
class TTAccClauseFinalize(TTAccClause):
    kind = "finalize"
class TTAccClauseIfPresent(TTAccClause):
    kind = "if_present"

class TTAccLoopParallelismClause(TTAccClause):
    """Base class for loop parallelism specifications."""   
    def _assign_fields(self, tokens):
        self.arg = tokens[0]
    @property
    def is_value_specified(self):
        return self.arg != None
    def child_nodes(self):
        yield self.arg
    
class TTAccClauseGang(TTAccLoopParallelismClause):
    kind = "gang"
class TTAccClauseWorker(TTAccLoopParallelismClause):
    kind = "worker"
class TTAccClauseVector(TTAccLoopParallelismClause):
    kind = "vector"

class TTAccResourceLimitClause(TTAccClause):
    """Base class for resource limit specifications."""   
    def _assign_fields(self,tokens):
        self.arg = tokens[0]
    def child_nodes(self):
        yield self.arg

class TTAccClauseNumGangs(TTAccResourceLimitClause):
    kind = "num_gangs"
class TTAccClauseNumWorkers(TTAccResourceLimitClause):
    kind = "num_workers"
class TTAccClauseVectorLength(TTAccResourceLimitClause):
    kind = "vector_length"

class TTAccClauseDeviceNum(TTAccClause):
    def _assign_fields(self,tokens):
        self.arg = tokens[0]
    def child_nodes(self):
        yield self.arg

class TTAccClauseDeviceType(TTAccClause):

    def _assign_fields(self, tokens):
        self.kind = "device_type"
        self.device_types = tokens
    @property
    def kind(self):
        return "device_type"

class TTAccConditionalClause(TTAccClause):
    """Base class."""   

    def _assign_fields(self, tokens):
        self.condition = tokens[0]
    def child_nodes(self):
        yield self.condition

class TTAccClauseIf(TTAccConditionalClause):
    kind = "if"

class TTAccClauseSelf(TTAccConditionalClause):
    kind = "self"

# Update clause

class TTAccUpdateTargetClause(TTAccClause):
    """Base class."""   
 
    def _assign_fields(self,tokens):
        self.args = tokens[0]
    def child_nodes(self):
        yield self.args
    @property
    def update_device(self):
        return not sef.update_host

class TTAccClauseUpdateSelf(TTAccUpdateTargetClause):
    kind = "self"

    @property
    def update_host(self):
        return True

class TTAccClauseDevice(TTAccUpdateTargetClause):
    kind = "device"
    
    @property
    def update_host(self):
        return False

# mapping clauses
class UnprocessedMappingClause(base.TTNode):
    """Dummy parse action that is re-translated
    into specialized nodes.
    """

    def _assign_fields(self, tokens):
        self.kind = tokens[0]
        self.vars = tokens[1]

class TTAccMappingClause(TTAccClause):
    """Base clase of specialized mapping nodes.
    """
    def __init__(self,var_list):
        TTAccClause.__init__(self)
        self.vars = var_list 

# TODO treat separately because of readonly modifier
class TTAccClauseCopyin(TTAccMappingClause):
    kind = "copyin"

class TTAccClauseCopy(TTAccMappingClause):
    kind = "copy"
class TTAccClauseCopyout(TTAccMappingClause):
    kind = "copyout"
class TTAccClauseCreate(TTAccMappingClause):
    kind = "create"
class TTAccClauseNoCreate(TTAccMappingClause):
    kind = "no_create"
class TTAccClausePresent(TTAccMappingClause):
    kind = "present"
class TTAccClauseDeviceptr(TTAccMappingClause):
    kind = "deviceptr"
class TTAccClauseAttach(TTAccMappingClause):
    kind = "attach"
class TTAccClausePrivate(TTAccMappingClause):
    kind = "private"
class TTAccClauseFirstPrivate(TTAccMappingClause):
    kind = "first_private"
class TTAccClauseUseDevice(TTAccMappingClause):
    kind = "use_device"

class TTAccClauseDefault(TTAccClause):

    def _assign_fields(self, tokens):
        self.kind  ="default"
        self.arg = tokens[0]
    @property
    def kind(self):
        return "default"

    @property
    def is_none(self):
        return self.arg.lower() == "none"
    
    @property
    def is_present(self):
        return self.arg.lower() == "present"

class TTAccClauseReduction(TTAccClause):

    def _assign_fields(self, tokens):
        self.op, self.vars = tokens

    def child_nodes(self):
        yield from self.vars

    @property
    def kind(self):
        return "reduction"

    #def reductions(self, converter=traversals.make_fstr):
    #    result = {}
    #    op = converter(self.operator)
    #    if converter == traversals.make_cstr:
    #        # post-process
    #        op = conv.get_operator_name(op.lower())
    #        # "+" "*" "max" "min" "iand" "ior" "ieor" ".and." ".or." ".eqv." ".neqv."
    #    result[op] = [traversals.make_fstr(var) for var in self.vars]
    #    return result

class TTAccClauseBind(TTAccClause):

    def _assign_fields(self, tokens):
        self.arg = tokens[0]
    
    @property
    def kind(self):
        return "bind"

class TTAccClauseTile(TTAccClause):

    def _assign_fields(self, tokens):
        self.args = tokens

    def child_nodes(self):
        yield from self.args   
 
    @property
    def kind(self):
        return "tile"

class TTAccClauseCollapse(TTAccClause):

    def _assign_fields(self, tokens):
        self.arg = tokens[0]

    def child_nodes(self):
        yield self.arg
    
    @property
    def kind(self):
        return "collapse"

class TTAccClauseWait(TTAccClause):

    def _assign_fields(self, tokens):
       self.args = tokens
    
    @property
    def kind(self):
        return "wait"

#    def expressions(self):
#        return [traversals.make_fstr(expr) for expr in self.expressions]

class TTAccClauseAsync(TTAccClause):

    def _assign_fields(self, tokens):
        self.expression = tokens[0]
    @property
    def kind(self):
        return "async"

class UnprocessedNoArgumentClause(base.TTNode):
    """Dummy parse action that is re-translated
    into specialized nodes.
    """
    def _assign_fields(self, tokens):
        self.kind=tokens[0]

def acc_clause_parse_action(tokens):
    """:Returns specialized clause instance
    for generic clause instance generated by pyparsing."""
    clause = tokens[0]    
    if isinstance(clause,UnprocessedNoArgumentClause)
        for cls in [TTAccClauseSeq,
                    TTAccClauseAuto,
                    TTAccClauseIndependent,
                    TTAccClauseRead,
                    TTAccClauseWrite,
                    TTAccClauseCapture,
                    TTAccClauseUpdate,
                    TTAccClauseNohost,
                    TTAccClauseFinalize,
                    TTAccClauseIfPresent]:
            if cls.kind == clause.kind:
                return cls([]) 
        assert False, "clause could not be classified"
    if isinstance(clause,UnprocessedMappingClause)
        for cls in [TTAccClauseCopy,
                    TTAccClauseCopyin,
                    TTAccClauseCopyout,
                    TTAccClauseCreate,
                    TTAccClauseNoCreate,
                    TTAccClausePresent,
                    TTAccClauseDeviceptr,
                    TTAccClauseAttach,
                    TTAccClauseFirstPrivate,
                    TTAccClauseUseDevice]:
            if cls.kind == clause.kind:
                return cls(clause.vars) 
    else:
        return clause

#
# Directives
#

class TTAccDirective(base.TTNode)

    def __init__(self,clauses):
        base.TTNode._init(self,clauses)

    def _assign_fields(self,clauses):
        self.clauses = clauses
        self._check_clauses()

    def child_nodes(self):
        """:note: might be overwritten, don't delete to walk_clauses."""
        yield from self.clauses

    def walk_clauses(self):
        yield from self.clauses

    def _is_legal_clause(self,clause):
        assert False, "must be implemented by subclass"

    def _is_unique_clause(self,clause):
        assert False, "must be implemented by subclass"
    
    def _is_unique_clause_per_device_type(self,clause):
        assert False, "must be implemented by subclass"
    
    def _may_follow_device_type_clause(self,clause):
        assert False, "must be implemented by subclass"

    def _check_clauses(self):
        """Checks if all entries of self.clauses are allowed
        for the given directive, that unique clauses do not appear
        twice, and that only certain clauses appear
        after device_type clauses.
        """
        i_device_type_clause = -1
        current_device_type_clause_types = []
        for i,clause in enumerate(self.clauses):
            if not self._is_legal_clause(clause):
                raise util.error.SyntaxError(
                 "clause '{0}' cannot be specified for directive '{1}'".format(
                    clause.kind
                    self.kind
                  )
                )
            if self._is_unique_clause(clause):    
                for prev_clause in self.clauses[:i]:
                    if type(clause) == type(prev_clause):
                        raise util.error.SyntaxError(
                         "clause '{0}' can only be specified once for directive '{1}'".format(
                            clause.kind,
                            self.kind
                          )
                        )
            if type(clause) == TTAccClauseDeviceType:
                if i_device_type_clause >= 0:
                    if i-i_device_type_clause == 1:
                        raise util.error.SyntaxError(
                         "'device_type' clause cannot be specified directly
                          after 'device_type' clause".format(
                            clause.kind,
                          )
                        )
                i_device_type_clause = i
                current_device_type_clause_types.clear()
            elif i_device_type_clause >= 0:
                if not self._may_follow_device_type_clause(clause):
                    raise util.error.SyntaxError(
                     "clause '{0}' cannot be specified after 'device_type' clause".format(
                        clause.kind,
                      )
                    )
            # :note: no device_type encountered yet means default device_type 
            # is used.
            if self._is_unique_clause_per_device_type(clause):
                if type(clause) in current_device_type_clause_types:
                    raise util.error.SyntaxError(
                     "clause '{}' can only be specified once per device type".format(
                        clause.kind,
                      )
                    )
                current_device_type_clause_types.append(type(clause))

    def _check_has_at_least_one_of(self,required_clauses):
        """Checks if at least one of the required clauses is specified.
        :note: Must be called explicitly by subclass constructor.
        :param required_clauses: tuple of classes of which one is required."""
        if len(required_clauses):
            for clause in self.walk_clauses():
                if isinstance(clause,required_clauses):
                    return
            # hasn't returned implies: no required clause found
            raise util.error.SyntaxError(
              "at least one {} clause".format(
                ", ".join(["'"+c+"'" for c in required_clauses])
              )
              + " must appear on '{}' directive".format(self.kind)
            )

# end directives

class TTAccEndDirective(TTAccDirective):
    pass

class TTAccEndParallel(TTAccEndDirective):
    kind = "end parallel"
class TTAccEndData(TTAccEndDirective):
    kind = "end data"
class TTAccEndHostData(TTAccEndDirective):
    kind = "end host_data"
class TTAccEndCache(TTAccEndDirective):
    kind = "end cache" 
class TTAccEndAtomic(TTAccEndDirective):
    kind = "end atomic"
class TTAccEndSerial(TTAccEndDirective):
    kind = "end serial"
class TTAccEndKernels(TTAccEndDirective):
    kind = "end kernels"
class TTAccEndParallelLoop(TTAccEndDirective):
    kind = "end parallel loop"
class TTAccEndKernelsLoop(TTAccEndDirective):
    kind = "end kernels loop"

# directives

class TTAccComputeConstruct(TTAccDirective):
    def __init__(self,clauses):
        TTAccDirective.__init__(self)
        self.clauses = clauses
    
    def _is_legal_clause(self,clause):
        """Default implementation matches that
        of acc serial construct."""
        return isinstance(clause,(
            TTAccClauseAsync,
            TTAccClauseWait,
            #TTAccClauseNumGangs, # only kernels, parallel
            #TTAccClauseNumWorkers,
            #TTAccClauseVectorLength,
            TTAccClauseDeviceType,
            TTAccClauseIf,
            TTAccClauseSelf,
            # TTAccClauseReduction, # only serial, parallel
            TTAccClauseCopy,
            TTAccClauseCopyin,
            TTAccClauseCopyout,
            TTAccClauseCreate,
            TTAccClauseNoCreate,
            TTAccClausePresent,
            TTAccClauseDeviceptr,
            TTAccClauseAttach,
            #TTAccClausePrivate, # only serial, parallel
            #TTAccClauseFirstPrivate,
            TTAccClauseDefault,
          ))

    def _is_unique_clause(self,clause):
        """Default implementation matches that
        of acc serial construct."""
        return isinstance(clause,(
            TTAccClauseIf,
            TTAccClauseSelf,
            TTAccClauseDefault,
          ))
    
    def _is_unique_clause_per_device_type(self,clause):
        """Default implementation matches that
        of acc serial construct."""
        return isinstance(clause,(
            TTAccClauseAsync, # not prescribed in standard
            TTAccClauseWait,
          ))
    
    def _may_follow_device_type_clause(self,clause):
        return self._is_unique_clause_per_device_type(clause)  

# compute constructs
class TTAccSerial(TTAccComputeConstruct):
    kind = "serial"
    
    def _is_legal_clause(self,clause):
        if not TTAccComputeConstruct._is_legal_clause(self,clause):
            return isinstance(clause,(
              TTAccClauseReduction,
              TTAccClausePrivate,
              TTAccClauseFirstPrivate,
            ))
        return True

    def _is_unique_clause(self,clause):
        return isinstance(clause,(
            TTAccClauseIf,
            TTAccClauseSelf,
            TTAccClauseDefault,
          ))

class TTAccParallel(TTAccComputeConstruct):
    kind = "parallel"
    
    def _is_legal_clause(self,clause):
        if not TTAccComputeConstruct._is_legal_clause(self,clause):
            return isinstance(clause,(
              TTAccClauseNumGangs,
              TTAccClauseNumWorkers,
              TTAccClauseVectorLength,
              TTAccClauseReduction,
              TTAccClausePrivate,
              TTAccClauseFirstPrivate
            ))
        return True
    
    def _is_unique_clause_per_device_type(self,clause):
        if not TTAccComputeConstruct._is_unique_clause_per_device_type(self,clause):
            return isinstance(clause,(
                    TTAccClauseNumGangs,
                    TTAccClauseNumWorkers,
                    TTAccClauseVectorLength))
        return True

class TTAccKernels(TTAccComputeConstruct):
    kind = "kernels"
    
    def _is_legal_clause(self,clause):
        if not TTAccComputeConstruct._is_legal_clause(self,clause):
            return isinstance(clause,(
              TTAccClauseNumGangs,
              TTAccClauseNumWorkers,
              TTAccClauseVectorLength,
            ))
        return True
    
    def _is_unique_clause_per_device_type(self,clause):
        if not TTAccComputeConstruct._is_unique_clause_per_device_type(self,clause):
            return isinstance(clause,(
                    TTAccClauseNumGangs,
                    TTAccClauseNumWorkers,
                    TTAccClauseVectorLength))
        return True

# within compute and combined construct
class TTAccLoop(TTAccDirective):
    kind = "loop"   
 
    def _is_legal_clause(self,clause):
        return isinstance(clause,(
          TTAccClauseCollapse,
          TTAccClauseGang,
          TTAccClauseWorker,
          TTAccClauseVector,
          TTAccClauseSeq,
          TTAccClauseIndependent,
          TTAccClauseAuto,
          TTAccClauseTile,
          TTAccClauseDeviceType,
          TTAccClausePrivate,
          TTAccClauseReduction,
        ))
    
    def _is_unique_clause(self,clause):
        return False
 
    def _is_unique_clause_per_device_type(self,clause):
        return isinstance(clause,(
          TTAccClauseCollapse,
          TTAccClauseGang,
          TTAccClauseWorker,
          TTAccClauseVector,
          TTAccClauseSeq,
          TTAccClauseIndependent,
          TTAccClauseAuto,
          TTAccClauseTile,
        ))
    
    def _may_follow_device_type_clause(self,clause):
        return self._is_unique_clause_per_device_type(clause)  

class TTAccAtomic(TTAccDirective):
    kind = "atomic"
    
    def _is_legal_clause(self,clause):
        return isinstance(clause,
          TTAccClauseUpdate,
          TTAccClauseCapture,
          TTAccClauseRead,
          TTAccClauseWrite
        )
   
     def _is_unique_clause(self,clause):
        return True 
    def _is_unique_clause_per_device_type(self,clause):
        return False # doesn't matter
    def _may_follow_device_type_clause(self,clause):
        return False # doesn't matter

# combined constructs
class TTAccParallelLoop(TTAccParallel,TTAccLoop):
    kind = "parallel loop"

    def _is_legal_clause(self,clause):
        return (TTAccParallel._is_legal_clause(self,clause)
               or TTAccLoop._is_legal_clause(self,clause)

    def _is_unique_clause(self,clause):
        return (TTAccParallel._is_unique_clause(self,clause)
               or TTAccLoop._is_unique_clause(self,clause)
    
    def _is_unique_clause_per_device_type(self,clause):
        return (TTAccParallel._is_unique_clause_per_device_type(self,clause)
               or TTAccLoop._is_unique_clause_per_device_type(self,clause))
    
    def _may_follow_device_type_clause(self,clause):
        return self._is_unique_clause_per_device_type(clause)  
    
class TTAccKernelsLoop(TTAccKernels,TTAccLoop):
    kind = "kernels loop"

    def _is_legal_clause(self,clause):
        return (TTAccKernels._is_legal_clause(self,clause)
               or TTAccLoop._is_legal_clause(self,clause)

    def _is_unique_clause(self,clause):
        return (TTAccKernels._is_unique_clause(self,clause)
               or TTAccLoop._is_unique_clause(self,clause)
    
    def _is_unique_clause_per_device_type(self,clause):
        return (TTAccKernels._is_unique_clause_per_device_type(self,clause)
               or TTAccLoop._is_unique_clause_per_device_type(self,clause))
    
    def _may_follow_device_type_clause(self,clause):
        return self._is_unique_clause_per_device_type(clause)  

# data and host_data environment
class TTAccData(TTAccDirective):
    kind = "data"
  
    def _is_legal_clause(self,clause):
        return isinstance(clause,(
            TTAccClauseIf,
            TTAccClauseAsync,
            TTAccClauseWait,
            TTAccClauseDeviceType,
            TTAccClauseCopy,
            TTAccClauseCopyin,
            TTAccClauseCopyout,
            TTAccClauseCreate,
            TTAccClauseNoCreate,
            TTAccClausePresent,
            TTAccClauseDeviceptr,
            TTAccClauseAttach,
            TTAccClauseDefault,
          ))

    def _is_unique_clause(self,clause):
        return isinstance(clause,(
            TTAccClauseIf,
            TTAccClauseDefault,
          ))
    
    def _is_unique_clause_per_device_type(self,clause):
        return isinstance(clause,(
            TTAccClauseAsync, # not prescribed in standard
            TTAccClauseWait,
          ))
    
    def _may_follow_device_type_clause(self,clause):
        return self._is_unique_clause_per_device_type(clause)  


# declare directive
class TTAccDeclare(TTAccDirective):
    kind = "declare"
   
    def __init__(self,clauses):
        TTAccDirective.__init__(self,clauses)
        self._legal_clauses = (
          TTAccClauseCopy,
          TTAccClauseCopyin,
          TTAccClauseCopyout,
          TTAccClauseCreate,
          TTAccClausePresent,
          TTAccClauseDeviceptr,
          TTAccClauseDeviceResident,
          TTAccClauseLink
        )
        self._check_has_at_least_one_of(
          self._legal_clauses
        )
  
    def _is_legal_clause(self,clause):
        return isinstance(clause,self._legal_clauses)
    def _is_unique_clause(self,clause):
        return False
    def _is_unique_clause_per_device_type(self,clause):
        return False
    def _may_follow_device_type_clause(self,clause):
        return self._is_unique_clause_per_device_type(clause)  

# host_data
class TTAccHostData(TTAccDirective):
    kind = "host_data"    

    def _is_legal_clause(self,clause):
        return isinstance(clause,(
            TTAccClauseUseDevice,
            TTAccClauseIf,
            TTAccClauseIfPresent,
          ))

    def _is_unique_clause(self,clause):
        return isinstance(clause,(
            TTAccClauseIf,
            TTAccClauseDefault,
          ))
    
    def _is_unique_clause_per_device_type(self,clause):
        return isinstance(clause,(
            TTAccClauseAsync, # not prescribed in standard
            TTAccClauseWait,
          ))
    
    def _may_follow_device_type_clause(self,clause):
        return self._is_unique_clause_per_device_type(clause)  

# executable directives
class TTAccEnterData(TTAccDirective):
    kind = "enter data"

    def __init__(self,clauses):
        TTAccDirective.__init__(self,clauses)
        self._check_has_at_least_one_of(
          (TTAccClauseCopyin,TTAccClauseCreate,TTAccClauseAttach)
        )
  
    def _is_legal_clause(self,clause):
        return isinstance(clause,(
            TTAccClauseIf,
            TTAccClauseAsync,
            TTAccClauseWait,
            TTAccClauseCopyin,
            TTAccClauseCreate,
            TTAccClauseAttach,
          ))

    def _is_unique_clause(self,clause):
        return isinstance(clause,(
            TTAccClauseIf,
            TTAccClauseAsync,
          ))
    
    def _is_unique_clause_per_device_type(self,clause):
        return False 
    def _may_follow_device_type_clause(self,clause):
        return self._is_unique_clause_per_device_type(clause)

class TTAccExitData(TTAccDirective):
    kind = "exit data"

    def __init__(self,clauses):
        TTAccDirective.__init__(self,clauses)
        self._check_has_at_least_one_of(
          (TTAccClauseCopyout,TTAccClauseDelete,TTAccClauseDetach)
        )

    def _is_legal_clause(self,clause):
        return isinstance(clause,(
            TTAccClauseIf,
            TTAccClauseAsync,
            TTAccClauseWait,
            TTAccClauseCopyout,
            TTAccClauseDelete,
            TTAccClauseDetach,
            TTAccClauseFinalize
          ))

    def _is_unique_clause(self,clause):
        return isinstance(clause,(
            TTAccClauseIf,
            TTAccClauseAsync,
            TTAccClauseFinalize,
          ))
    
    def _is_unique_clause_per_device_type(self,clause):
        return False 
    def _may_follow_device_type_clause(self,clause):
        return self._is_unique_clause_per_device_type(clause)

class TTAccInit(TTAccDirective):
    kind = "init"

    def _is_legal_clause(self,clause):
        return isinstance(clause,(
            TTAccClauseDeviceType,
            TTAccClauseDeviceNum,
            TTAccClauseIf,
          ))

    def _is_unique_clause(self,clause):
        return isinstance(clause,(
            TTAccClauseIf
          ))
    
    def _is_unique_clause_per_device_type(self,clause):
        return isinstance(clause,(
            TTAccClauseDeviceNum,
          ))
    def _may_follow_device_type_clause(self,clause):
        return self._is_unique_clause_per_device_type(clause)

class TTAccShutdown(TTAccDirective):
    kind = "shutdown"

    def _is_legal_clause(self,clause):
        return isinstance(clause,(
            TTAccClauseDeviceType,
            TTAccClauseDeviceNum,
            TTAccClauseIf,
          ))

    def _is_unique_clause(self,clause):
        return isinstance(clause,(
            TTAccClauseIf
          ))
    
    def _is_unique_clause_per_device_type(self,clause):
        return isinstance(clause,(
            TTAccClauseDeviceNum,
          ))
    def _may_follow_device_type_clause(self,clause):
        return self._is_unique_clause_per_device_type(clause)

class TTAccSet(TTAccDirective):
    kind = "set"

    def _is_legal_clause(self,clause):
        return isinstance(clause,(
            TTAccClauseDeviceType,
            TTAccClauseDeviceNum,
            TTAccClauseIf,
          ))

    def _is_unique_clause(self,clause):
          """2.14.3 Restrictions:
          Two instances of the same clause may not appear on the same directive.
          """
          return self._is_legal_clause(self,clause)    
    def _is_unique_clause_per_device_type(self,clause):
        return False # all clauses are unique
    def _may_follow_device_type_clause(self,clause):
        return self._is_unique_clause_per_device_type(clause)

class TTAccUpdate(TTAccDirective):
    kind = "update"
    
    def __init__(self,clauses):
        TTAccDirective.__init__(self,clauses)
        self._check_has_at_least_one_of(
          (TTAccClauseUpdateSelf,TTAccClauseUpdateDevice)
        )

    def _is_legal_clause(self,clause):
        return isinstance(clause,(
            TTAccClauseAsync,
            TTAccClauseWait,
            TTAccClauseDeviceType,
            TTAccClauseIf,
            TTAccClauseIfPresent,
            TTAccClauseUpdateSelf,
            TTAccClauseUpdateDevice,
          ))

    def _is_unique_clause(self,clause):
        return self._is_legal_clause(self,clause)    
    def _is_unique_clause_per_device_type(self,clause):
        """2.14.4 Restrictions: 
        Only the async and wait clauses may follow a device_type clause."""
        return isinstance(clause,(
            TTAccClauseAsync,
            TTAccClauseWait,
          ))
    def _may_follow_device_type_clause(self,clause):
        return self._is_unique_clause_per_device_type(clause)
     
class TTAccWait(TTAccDirective):
    kind = "wait"

class TTAccCache(TTAccDirective):
    kind = "cache"

class UnprocessedGenericDirective(base.TTNode):
    """Dummy parse action that is re-translated
    into specialized nodes.
    """
    def _assign_fields(self,tokens):
        self.kind = util.parsing.tokenize(tokens[0])
        self.clauses = tokens[1] 

class UnprocessedEndDirective(base.TTNode):
    """Dummy parse action that is translated
    into specialized nodes.
    """
    def _assign_fields(self,tokens):
        self.kind = util.parsing.tokenize(tokens[0])

def acc_directive_parse_action(tokens):
    directive = tokens[0]
    if isinstance(directive,UnprocessedEndDirective)
        for cls in [TTAccEndParallel,
                    TTAccEndData,
                    TTAccEndHostData,
                    TTAccEndCache,
                    TTAccEndAtomic,
                    TTAccEndSerial,
                    TTAccEndKernels,
                    TTAccEndParallelLoop,
                    TTAccEndKernelsLoop]:
        if directive.kind == util.tokenize(cls.kind):
            return cls()
        assert False, "could not be classified"
    elif isinstance(UnprocessedGenericDirective):
        assert False, "could not be classified"
        pass
    
def set_acc_parse_actions(grammar):
    grammar.acc_clause_gang.setParseAction(TTAccClauseGang)
    grammar.acc_clause_worker.setParseAction(TTAccClauseWorker)
    grammar.acc_clause_vector.setParseAction(TTAccClauseVector)
    grammar.acc_clause_num_gangs.setParseAction(TTAccClauseNumGangs)
    grammar.acc_clause_num_workers.setParseAction(TTAccClauseNumWorkers)
    grammar.acc_clause_vector_length.setParseAction(TTAccClauseVectorLength)
    grammar.acc_clause_device_type.setParseAction(TTAccClauseDeviceType)
    grammar.acc_clause_if.setParseAction(TTAccClauseIf)
    grammar.acc_clause_default.setParseAction(TTAccClauseDefault)
    grammar.acc_clause_collapse.setParseAction(TTAccClauseCollapse)
    grammar.acc_clause_self.setParseAction(TTAccClauseSelf)
    grammar.acc_clause_bind.setParseAction(TTAccClauseBind)
    grammar.acc_clause_reduction.setParseAction(TTAccClauseReduction)
    grammar.acc_clause_tile.setParseAction(TTAccClauseTile)
    grammar.acc_clause_wait.setParseAction(TTAccClauseWait)
    grammar.acc_clause_async.setParseAction(TTAccClauseAsync)
    grammar.acc_mapping_clause.setParseAction(TTAccMappingClause)
    grammar.acc_noarg_clause.setParseAction(UnprocessedNoArgumentClause)
    grammar.acc_clause.setParseAction(acc_clause_parse_action)
    # 
    grammar.acc_generic_directive.setParseAction(UnprocessedGenericDirective)
    grammar.acc_end_directive.setParseAction(UnprocessedEndDirective)
    grammar.acc_directive.setParseAction(acc_directive_parse_action)
    # directives with (optional) argument list
    # todo:
    #grammar.acc_wait.setParseAction(TTAccWait)
    #grammar.acc_cache.setParseAction(TTAccCache)
    #grammar.acc_routine.setParseAction(TTAccRoutine)
    #grammar.acc_directive.setParseAction(acc_directive_parse_action)
