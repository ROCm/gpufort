
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
        if len(tokens):
            self.arg = tokens[0]
        else:
            self.arg = None
    @property
    def arg_specified(self):
        return self.arg != None
    def child_nodes(self):
        if self.arg_specified:
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
    kind = "device_num"

    def _assign_fields(self,tokens):
        self.arg = tokens[0]

    def child_nodes(self):
        yield self.arg

class TTAccClauseDeviceType(TTAccClause):
    kind = "device_type"

    def _assign_fields(self, tokens):
        self.device_types = tokens

    def named_device_types(self,lower_case=True):
        """:return: All arguments that are identifiers, i.e. not '*'.
        :param lower_case: Convert to lower case, defaults to True.
        :see: has_named_device_types
        """ 
        result = []
        for device_type in ttnode.device_types:
            if device_type.isidentifier():
                if lower_case:
                    result.append(device_type.lower())
                else:
                    result.append(device_type)
        return device_type

    @property
    def has_named_device_types(self):
        """:return: If some arguments are identifiers, i.e. not '*'.
        """
        for device_type in ttnode.device_types:
            if device_type.isidentifier():
                return True
        return False

class TTAccClauseIf(TTAccClause):
    kind = "if"    

    def _assign_fields(self, tokens):
        self.arg = tokens[0]
    def child_nodes(self):
        yield self.arg

class TTAccClauseSelf(TTAccClause):
    kind = "self"
    
    def _assign_fields(self, tokens):
        self.arg = tokens[0]
    def child_nodes(self):
        yield self.arg

# Update clause

class TTAccClauseUpdateSelf(TTAccClause):
    """:note: Is called UpdateSelf to distinguish from compute
    constructs Self clause."""
    kind = "self"
    
    def _assign_fields(self,tokens):
        self.args = tokens
    def child_nodes(self):
        yield from self.args

class TTAccClauseUpdateDevice(TTAccClause):
    kind = "device"
    
    def _assign_fields(self,tokens):
        self.args = tokens
    def child_nodes(self):
        yield from self.args

# mapping clauses
class UnprocessedGenericMappingClause(base.TTNode):
    """Dummy parse action that is re-translated
    into specialized nodes.
    """

    def _assign_fields(self, tokens):
        self.kind = tokens[0]
        self.var_list = tokens[1]

class TTAccMappingClause(TTAccClause):
    """Base clase of specialized mapping nodes.
    """
    def __init__(self,var_list):
        TTAccClause.__init__(self)
        self.var_list = var_list 

    def child_nodes(self):
        yield from self.var_list

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

class TTAccClauseCopyin(TTAccMappingClause):
    """:note: Handled separately because of readonly modifier."""
    kind = "copyin"

    def __init__(self,tokens):
        TTAccMappingClause.__init__(self,tokens[-1])
        if len(tokens) == 2:
            assert tokens[0].lower() == "readonly"
            self.readonly = True
        else:
            self.readonly = False

class TTAccClauseDefault(TTAccClause):
    kind = "default"

    def _assign_fields(self, tokens):
        self.arg = tokens[0]

    @property
    def is_none(self):
        return self.arg.lower() == "none"
    
    @property
    def is_present(self):
        return self.arg.lower() == "present"

class TTAccClauseReduction(TTAccClause):
    kind = "reduction"

    def _assign_fields(self, tokens):
        self.op = tokens[0]
        self.var_list = list(tokens[1:])

    def child_nodes(self):
        yield from self.var_list

    #def reductions(self, converter=traversals.make_fstr):
    #    result = {}
    #    op = converter(self.operator)
    #    if converter == traversals.make_cstr:
    #        # post-process
    #        op = conv.get_operator_name(op.lower())
    #        # "+" "*" "max" "min" "iand" "ior" "ieor" ".and." ".or." ".eqv." ".neqv."
    #    result[op] = [traversals.make_fstr(var) for var in self.var_list]
    #    return result

class TTAccClauseBind(TTAccClause):
    kind = "bind"

    def _assign_fields(self, tokens):
        self.arg = tokens[0]

class TTAccClauseTile(TTAccClause):
    kind = "tile"

    def _assign_fields(self, tokens):
        self.args = list(tokens)

    def child_nodes(self):
        yield from self.args   

class TTAccClauseCollapse(TTAccClause):
    kind = "collapse"

    def _assign_fields(self, tokens):
        self.arg = tokens[-1]
        if len(tokens) == 2:
            assert tokens[0].lower() == "force"
            self.force = True
        else:
            self.force = False

    def child_nodes(self):
        yield self.arg

class TTAccClauseWait(TTAccClause):
    kind = "wait"

    def _assign_fields(self, tokens):
       if len(tokens) == 1:
            self.devnum = None
            self.queues = list(tokens[0])
       elif len(tokens) == 2:
            self.devnum = tokens[0]
            self.queues = list(tokens[1])
       else:
           self.devnum = None
           self.queues = []

    @property
    def devnum_specified(self):
        return self.devnum != None
    
    @property
    def queues_specified(self):
        return len(self.queues)
    
    def child_nodes(self):
        yield from self.queues

#    def expressions(self):
#        return [traversals.make_fstr(expr) for expr in self.expressions]

class TTAccClauseAsync(TTAccClause):
    kind = "async"

    def _assign_fields(self, tokens):
        self.queues = list(tokens)

    @property
    def queues_specified(self):
        return len(queues)

class UnprocessedNoArgumentClause(base.TTNode):
    """Dummy parse action that is re-translated
    into specialized nodes.
    """
    def _assign_fields(self, tokens):
        self.kind=tokens[0]

# artificial clauses for routine and cache
class TTAccArtificialClauseRoutine(TTAccClause):
    """Artificial routine clause for routine directive."""   
 
    def _assign_fields(self,tokens):
        if len(tokens):
            self.name = tokens[0]
        else:
            self.name = None
  
    @property
    def name_specified(self):
        return self.name != None

class TTAccArtificialClauseCache(TTAccClause):
    """Artificial cache clause for cache directive."""   
 
    def _assign_fields(self,tokens):
        self.var_list = list(tokens[-1])
        if len(tokens) == 2:
            assert tokens[0].lower() == "readonly"
            self.readonly = True
        else:
            self.readonly = False

def acc_clause_parse_action(tokens):
    """:Returns specialized clause instance
    for generic clause instance generated by pyparsing."""
    clause = tokens[0]    
    if isinstance(clause,UnprocessedNoArgumentClause):
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
    if isinstance(clause,UnprocessedGenericMappingClause):
        for cls in [TTAccClauseCopy,
                    #TTAccClauseCopyin, handled separately because of readonly modifier
                    TTAccClauseCopyout,
                    TTAccClauseCreate,
                    TTAccClauseNoCreate,
                    TTAccClausePresent,
                    TTAccClauseDeviceptr,
                    TTAccClauseAttach,
                    TTAccClauseFirstPrivate,
                    TTAccClauseUseDevice]:
            if cls.kind == clause.kind:
                return cls(clause.var_list) 
    else:
        return clause

#
# Directives
#

class TTAccDirective(base.TTStatement):

    def __init__(self,clauses):
        base.TTNode._init(self)
        self.clauses = clauses
        self._check_clauses()
        self._named_device_types = self._get_named_device_types()

    def child_nodes(self):
        """:note: might be overwritten, don't delete to walk_clauses."""
        yield from self.clauses

    def walk_clauses(self):
        yield from self.clauses

    def walk_matching_clauses(self,cls):
        """:return: All clauses that are an instance of 'cls'.
        :param str cls: The class or base class of the clause to lookup.
        """
        for clause in self.walk_clauses():
            if isinstance(clause,cls):
                yield cls
 
    def _get_named_device_types(self):
        """:return: A list of device type identifiers found 
                    in the device type clauses.
        :note: Converts all identifiers to lower case.
        """
        named_device_types = []

        def already_detected_(device_type):
            nonlocal named_device_types
                    
        for clause in self.walk_matching_clauses(TTAccClauseDeviceType):
            if clause.has_named_device_types:
                for device_type in clause.named_device_types:
                    for other in named_device_types:
                        if other.lower() == device_type.lower():
                            return util.error.SyntaxError(
                              "device type '{}' specified more than once in ".format(device_type)
                              + "'device_type' clauses"
                            )
                    named_device_types.append(device_type.lower())
        return named_device_types 

    def walk_clauses_device_type(self,device_type):
        """:return: Walk clauses that apply to the given device type.
        :param str device_type: Device type such as 'nvidia', 'radeon', or None (default).
        :param str cls: The class or base class of the clause to lookup.
        """
        result = []
        current_device_types = [] 
        def clause_applies_to_device_type_(device_type):
            nonlocal current_device_types
            return (
              not len(current_device_types) # no device_type clause encountered yet
              or device_type in current_device_types # device-type-specific clauses found
              or current_device_types[0] == "*"
            )
        for clause in self.walk_clauses():
            if isinstance(clause,TTAccClauseDeviceType):
                current_device_types = clause.device_types
            elif isinstance(clause,cls):
                if clause_applies_to_device_type_(device_type):
                    yield clause

    def _is_legal_clause(self,clause):
        assert False, "must be implemented by subclass"

    def _is_unique_clause(self,clause):
        assert False, "must be implemented by subclass"
    
    def _is_unique_clause_per_device_type(self,clause):
        assert False, "must be implemented by subclass"
    
    def _may_follow_device_type_clause(self,clause):
        return self._is_unique_clause_per_device_type(clause)
    
    def _check_clauses(self):
        """Checks if all entries of self.clauses are allowed
        for the given directive, that unique clauses do not appear
        twice, and that only certain clauses appear
        after device_type clauses.
        """
        i_device_type_clause = -1
        current_device_type_clause_types = []
        for i,clause in enumerate(self.clauses):
            assert isinstance(clause,TTAccClause), "clause "+str(i)+" is not of type TTAccClause, is of type "+str(type(clause))
            if not self._is_legal_clause(clause):
                raise util.error.SyntaxError(
                 "clause '{0}' cannot be specified for directive '{1}'".format(
                    clause.kind,
                    self.kind
                  )
                )
            #
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
                         "'device_type' clause cannot be specified directly"+
                         "after 'device_type' clause"
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
              "at least one {}, or '{}' clause".format(
                ", ".join(["'"+c.kind+"'" for c in required_clauses[:-1]]),
                required_clauses[-1].kind
              )
              + " must appear on '{}' directive".format(self.kind)
            )

# end directives

class TTAccEndDirective(TTAccDirective):
    
    def _is_legal_clause(self,clause):
        return False 

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

class TTAccConstruct(TTAccDirective,base.TTContainer):
    
    def __init__(self,clauses):
        base.TTContainer._init(self)
        TTAccDirective.__init__(self,clauses)

class TTAccComputeConstruct(TTAccConstruct):
    
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

    def _contains_single_assignment_statement(self):
        if len(self.body) == 1:
            return isinstance(self.body[0],tree.TTAssignment)
        return False

    @property
    def is_device_to_device_copy(self):
        if self._contains_single_assignment_statement():
            return self.body[0]
        return False 

# within compute and combined construct
class TTAccLoop(TTAccConstruct):
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

# combined constructs
class TTAccParallelLoop(TTAccParallel,TTAccLoop):
    kind = "parallel loop"

    def _is_legal_clause(self,clause):
        return (TTAccParallel._is_legal_clause(self,clause)
               or TTAccLoop._is_legal_clause(self,clause))

    def _is_unique_clause(self,clause):
        return (TTAccParallel._is_unique_clause(self,clause)
               or TTAccLoop._is_unique_clause(self,clause))
    
    def _is_unique_clause_per_device_type(self,clause):
        return (TTAccParallel._is_unique_clause_per_device_type(self,clause)
               or TTAccLoop._is_unique_clause_per_device_type(self,clause))
    
    
class TTAccKernelsLoop(TTAccKernels,TTAccLoop):
    kind = "kernels loop"

    def _is_legal_clause(self,clause):
        return (TTAccKernels._is_legal_clause(self,clause)
               or TTAccLoop._is_legal_clause(self,clause))

    def _is_unique_clause(self,clause):
        return (TTAccKernels._is_unique_clause(self,clause)
               or TTAccLoop._is_unique_clause(self,clause))
    
    def _is_unique_clause_per_device_type(self,clause):
        return (TTAccKernels._is_unique_clause_per_device_type(self,clause)
               or TTAccLoop._is_unique_clause_per_device_type(self,clause))
    

# data and host_data environment
class TTAccData(TTAccDirective):
    kind = "data"
    
    def __init__(self,clauses):
        """note: 2.6.5 Restrictions: At least one copy, copyin, copyout, create, no_create, present, deviceptr
           attach, or default clause must appear on a data construct."""
        TTAccDirective.__init__(self,clauses)
        self._check_has_at_least_one_of((
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
          return self._is_legal_clause(clause)    
    def _is_unique_clause_per_device_type(self,clause):
        return False # all clauses are unique

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
        return  
    def _is_unique_clause_per_device_type(self,clause):
        """2.14.4 Restrictions: 
        Only the async and wait clauses may follow a device_type clause."""
        return isinstance(clause,(
            TTAccClauseAsync,
            TTAccClauseWait,
          ))
     
class TTAccWait(TTAccDirective):
    kind = "wait"
    
    def __init__(self,tokens):
        assert isinstance(tokens[0],TTAccClauseWait)
        TTAccDirective.__init__(self,tokens)
   
    def _is_legal_clause(self,clause):
        return isinstance(clause,(
            TTAccClauseWait,
            TTAccClauseAsync,
            TTAccClauseIf,
          ))

    def _is_unique_clause(self,clause):
        return isinstance(clause,(
            TTAccClauseWait,
            TTAccClauseIf,
          ))
    def _is_unique_clause_per_device_type(self,clause):
        return False

class TTAccRoutine(TTAccDirective):
    kind = "routine"
    
    def __init__(self,tokens):
        assert isinstance(tokens[0],TTAccArtificialClauseRoutine)
        self._arg_clause = tokens[0]
        TTAccDirective.__init__(self,tokens[1:])
        self._check_has_at_least_one_of((
          TTAccClauseGang,
          TTAccClauseWorker,
          TTAccClauseVector,
          TTAccClauseSeq
        ))

    @property
    def name_specified(self):
        return self._arg_clause.name_specified
    
    @property
    def name(self):
        return self._arg_clause.name
   
    def _is_legal_clause(self,clause):
        return isinstance(clause,(
            TTAccClauseGang, # :todo: TTAccRoutineGang to support dim
            TTAccClauseWorker,
            TTAccClauseVector,
            TTAccClauseSeq,
            TTAccClauseBind, # :todo: identifier (as in language being compiled) vs string (unmodified)
            TTAccClauseDeviceType,
            TTAccClauseDeviceNoHost,
          ))

    def _is_unique_clause(self,clause):
        return isinstance(clause,(
            TTAccClauseBind,
            TTAccClauseIf,
          ))
    def _is_unique_clause_per_device_type(self,clause):
        return isinstance(clause,(
            TTAccClauseGang, # :todo: TTAccRoutineGang to support dim
            TTAccClauseWorker,
            TTAccClauseVector,
            TTAccClauseSeq,
            TTAccClauseBind, # :todo: identifier (as in language being compiled) vs string (unmodified)
          ))


class TTAccCache(TTAccDirective):
    kind = "cache"

    def __init__(self,tokens):
        assert isinstance(tokens[0],TTAccArtificialClauseCache)
        self._arg_clause = tokens[0]
        TTAccDirective.__init__(self,[]) # no clauses
    
    @property
    def var_list(self):
        return self._arg_clause.var_list
        
    def _is_legal_clause(self,clause):
        return False
    def _is_unique_clause(self,clause):
        return False
    def _is_unique_clause_per_device_type(self,clause):
        return False

class UnprocessedGenericDirective(base.TTNode):
    """Dummy parse action that is re-translated
    into specialized nodes.
    """
    def _assign_fields(self,tokens):
        self.kind = tokens[0]
        self.clauses = tokens[1:]

class UnprocessedEndDirective(base.TTNode):
    """Dummy parse action that is translated
    into specialized nodes.
    """
    def _assign_fields(self,tokens):
        self.kind = tokens[0]

def acc_directive_parse_action(tokens):
    directive = tokens[0]
    if isinstance(directive,UnprocessedEndDirective):
        for cls in [
            TTAccEndParallel,
            TTAccEndData,
            TTAccEndHostData,
            TTAccEndCache,
            TTAccEndAtomic,
            TTAccEndSerial,
            TTAccEndKernels,
            TTAccEndParallelLoop,
            TTAccEndKernelsLoop
          ]:
            if util.parsing.tokenize(directive.kind) == util.parsing.tokenize(cls.kind):
                return cls()
        assert False, "could not classify end directive"
    elif isinstance(directive,UnprocessedGenericDirective):
        for cls in [
            TTAccData,
            TTAccEnterData,
            TTAccExitData,
            TTAccHostData,
            TTAccLoop,
            TTAccAtomic,
            TTAccDeclare,
            TTAccInit,
            TTAccSet,
            TTAccUpdate,
            TTAccParallelLoop,
            TTAccKernelsLoop,
            TTAccSerial,
            TTAccParallel,
            TTAccKernels,
            TTAccShutdown,
          ]:
            if util.parsing.tokenize(directive.kind) == util.parsing.tokenize(cls.kind):
                return cls(directive.clauses)
        assert False, "could not classify generic directive" # no exception as grammar prevents other directive kinds
    elif isinstance(directive,(
         TTAccWait,
         TTAccRoutine,
         TTAccCache,
         TTAccUpdate,
      )):
        return directive
    
def set_acc_parse_actions(grammar):
    """Register parse actions for grammar nodes.
    :note: Many clauses are initially parsed
           and instantiated as generic 
           `UnprocessedNoArgumentClause` or `UnprocessedGenericMappingClause` objects.
           Then, they are post-processed and converted into
           a specialized class per clause via `acc_clause_parse_action`.
           All parse actions are invoked by pyparsing in the correct order.
           Notable exceptions are clauses with optional arguments. Those
           are treated individually.
    :note: Most directives are initially parsed and instantiated as 
           generic `UnprocessedGenericDirective` or `UnprocessedEndDirective` objects.
           Then,they are post-processed and converted into
           a specialized class per directive.
           The only exceptions are the classes TTAccWait, TTAccRoutine, and
           TTAccCache as they have a unique structure (required or optional argument).
           All parse actions are invoked by pyparsing in the correct order.
    """
    grammar.acc_clause_gang.setParseAction(TTAccClauseGang)
    grammar.acc_clause_worker.setParseAction(TTAccClauseWorker)
    grammar.acc_clause_vector.setParseAction(TTAccClauseVector)
    grammar.acc_clause_num_gangs.setParseAction(TTAccClauseNumGangs)
    grammar.acc_clause_num_workers.setParseAction(TTAccClauseNumWorkers)
    grammar.acc_clause_vector_length.setParseAction(TTAccClauseVectorLength)
    grammar.acc_clause_device_type.setParseAction(TTAccClauseDeviceType)
    grammar.acc_clause_if.setParseAction(TTAccClauseIf)
    grammar.acc_clause_self.setParseAction(TTAccClauseSelf)
    grammar.acc_clause_default.setParseAction(TTAccClauseDefault)
    grammar.acc_clause_collapse.setParseAction(TTAccClauseCollapse)
    grammar.acc_clause_update_self.setParseAction(TTAccClauseUpdateSelf)
    grammar.acc_clause_update_device.setParseAction(TTAccClauseUpdateDevice)
    grammar.acc_clause_bind.setParseAction(TTAccClauseBind)
    grammar.acc_clause_reduction.setParseAction(TTAccClauseReduction)
    grammar.acc_clause_tile.setParseAction(TTAccClauseTile)
    grammar.acc_clause_wait.setParseAction(TTAccClauseWait)
    grammar.acc_clause_async.setParseAction(TTAccClauseAsync)
    grammar.acc_clause_copyin.setParseAction(TTAccClauseCopyin)
    grammar.acc_generic_mapping_clause.setParseAction(UnprocessedGenericMappingClause)
    grammar.acc_noarg_clause.setParseAction(UnprocessedNoArgumentClause)
    grammar.acc_artificial_clause_routine.setParseAction(TTAccArtificialClauseRoutine)
    grammar.acc_artificial_clause_cache.setParseAction(TTAccArtificialClauseCache)
    #
    grammar.acc_clause.setParseAction(acc_clause_parse_action)
    # 
    grammar.acc_generic_directive.setParseAction(UnprocessedGenericDirective)
    grammar.acc_end_directive.setParseAction(UnprocessedEndDirective)
    # directives with (optional) argument list
    grammar.acc_wait.setParseAction(TTAccWait)
    grammar.acc_cache.setParseAction(TTAccCache)
    grammar.acc_routine.setParseAction(TTAccRoutine)
    grammar.acc_update.setParseAction(TTAccUpdate)
    #
    grammar.acc_directive.setParseAction(acc_directive_parse_action)
