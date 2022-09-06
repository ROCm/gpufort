# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import util

from .. import opts
from .. import conv

from . import grammar
from . import base
from . import directives

class TTAccClauseGang(TTNode):
    def _assign_fields(self, tokens):
        self.kind = "gang"
        self.value = tokens[0]

class TTAccClauseWorker(TTNode):
    def _assign_fields(self, tokens):
        self.kind = "worker"
        self.value = tokens[0]

class TTAccClauseVector(TTNode):
    def _assign_fields(self, tokens):
        self.kind = "vector"
        self.value = tokens[0]

class TTAccClauseNumGangs(TTNode):
    def _assign_fields(self, tokens):
        self.kind = "num_gangs"
        self.value = tokens[0]

class TTAccClauseNumWorkers(TTNode):
    def _assign_fields(self, tokens):
        self.kind = "num_workers"
        self.value = tokens[0]

class TTAccClauseVectorLength(TTNode):
    def _assign_fields(self, tokens):
        self.kind = "vector_length"
        self.value = tokens[0]

class TTAccClauseDeviceType(base.TTNode):

    def _assign_fields(self, tokens):
        self.kind = "device_type"
        self.device_types = tokens

    def device_types(self):
        return self.device_types


class TTAccClauseIf(base.TTNode):

    def _assign_fields(self, tokens):
        self.kind="if"
        self.condition = tokens[0]

    def condition(self):
        return base.make_f_str(self.condition)


class TTAccClauseSelf(base.TTNode):

    def _assign_fields(self, tokens):
        self.kind="self"
        self.condition = tokens[0]

    def condition(self):
        return base.make_f_str(self.condition)


class TTAccMappingClause(base.TTNode):

    def _assign_fields(self, tokens):
        self.kind = tokens[0]
        self.var_list = tokens[1].asList()

    def var_names(self, converter=base.make_f_str):
        return [var.identifier_part(converter) for var in self.var_list]

    def var_expressions(self, converter=base.make_f_str):
        return [converter(var) for var in self.var_list]


class TTAccClauseDefault(base.TTNode):

    def _assign_fields(self, tokens):
        self.kind="default"
        self.value = tokens


class TTAccClauseReduction(base.TTNode):

    def _assign_fields(self, tokens):
        self.kind="reduction"
        self.operator, self.vars = tokens

    def reductions(self, converter=base.make_f_str):
        result = {}
        op = converter(self.operator)
        if converter == base.make_c_str:
            # post-process
            op = conv.get_operator_name(op.lower())
            # "+" "*" "max" "min" "iand" "ior" "ieor" ".and." ".or." ".eqv." ".neqv."
        result[op] = [base.make_f_str(var) for var in self.vars]
        return result


class TTAccClauseBind(base.TTNode):

    def _assign_fields(self, tokens):
        self.kind="bind"
        pass


class TTAccClauseTile(base.TTNode):

    def _assign_fields(self, tokens):
        self.kind="tile"
        self.tiles_per_dim = tokens[0]

    def values():
        return self.tiles_per_dim


class TTAccClauseCollapse(base.TTNode):

    def _assign_fields(self, tokens):
        self.kind="collapse"
        self._value = tokens[0]

    def value(self):
        return int(self._value._value)


class TTAccClauseWait(base.TTNode):

    def _assign_fields(self, tokens):
        self.kind="wait"
        if tokens != None:
            try:
                self.expressions = tokens[0].asList()
            except AttributeError:
                if isinstance(tokens[0],list):
                    self.expressions = tokens[0]
                else:
                    raise

    def expressions(self):
        return [base.make_f_str(expr) for expr in self.expressions]


class TTAccClauseAsync(base.TTNode):

    def _assign_fields(self, tokens):
        self.kind="async"
        self.expression = tokens[0]

#
# Directives
#
class TTAccDirectiveBase(base.TTNode):

    def _assign_fields(self, tokens):
        self.clauses = tokens[0]

    def handle_mapping_clause(self, clause_kinds, converter=base.make_f_str):
        result = []
        for clause in self.clauses:
            if type(clause) == TTAccMappingClause and\
                    clause.kind in clause_kinds:
                result += clause.var_expressions()
        return result

    def _format(self, f_snippet):
        return directives.format_directive(f_snippet, 80)

    def omp_f_str(self):
        assert False, "Not implemented"


# TODO add host data
class TTAccDataManagementDirectiveBase(TTAccDirectiveBase):
    """
    Abstract class that handles the following clauses
      if( condition )
      copy( var-list )
      copyin( [readonly:]var-list )
      copyout( var-list )
      create( var-list )
      no_create( var-list )
      present( var-list )
      deviceptr( var-list )
      attach( var-list )
      default( none | present )
    which are common to data to parallel, kernels and data directives.
    
    Further handles the following additional clauses
    that can appear with 'data', 'enter data' and 'exit data'
    directives:
      if( condition )  
      async [( int-expr )]
      wait [( wait-argument )]
      finalize
    """

    def async_nowait(self):
        clause = base.find_first(self.clauses, TTAccClauseAsync)
        if not clause is None:
            return base.make_f_str(clause.expression)
        else:
            return grammar.CLAUSE_NOT_FOUND

    def deviceptrs(self):
        return self.handle_mapping_clause(["deviceptr"])

    def create_alloc_vars(self):
        return self.handle_mapping_clause(["create"])

    def no_create_vars(self):
        return self.handle_mapping_clause(["no_create"])

    def present_vars(self):
        return self.handle_mapping_clause(["present"])

    def copy_map_to_from_vars(self):
        return self.handle_mapping_clause(["copy"])

    def copyin_map_to_vars(self):
        return self.handle_mapping_clause(["copyin"])

    def copyout_map_from_vars(self):
        return self.handle_mapping_clause(["copyout"])

    def attach_vars(self):
        return self.handle_mapping_clause(["attach"])

    def detach_vars(self):
        return self.handle_mapping_clause(["detach"])

    def delete_release_vars(self):
        return self.handle_mapping_clause(["delete"])

    def if_condition(self):
        clause = base.find_first(self.clauses, TTAccClauseIf)
        if not clause is None:
            return [base.make_f_str(e) for e in clause.condition()]
        else:
            return ""

    def self_condition(self):
        clause = base.find_first(self.clauses, TTAccClauseSelf)
        if not clause is None:
            return [base.make_f_str(e) for e in clause.expression()]
        else:
            return ""

    def present_by_default(self):
        clause = base.find_first(self.clauses, TTAccClauseDefault)
        if not clause is None:
            return clause.value == "present"
        else:
            return True

    def omp_f_str(self,
                  arrays_in_body=[],
                  inout_arrays_in_body=[],
                  depend={},
                  prefix=""):
        # TODO split in frontend and backend?
        """
        :param: arrays_in_body      all array lvalues and rvalues
        :param: inout_arrays_in_body all used in embedded regions whose elements appear as lvalues
        :return: 'prefix' [mapped-clauses]
        
        where the clauses are mapped as follows:
        
        attach,copyin:  "map(to: var-list)"
        detach,copyout: "map(from: var-list)"
        copy:           "map(tofrom: var-list)"
        create:         "map(alloc: var-list)"
        delete:         "map(release: var-list)"
        present:        "" 
        no_create:      ""
        """

        to_list = self.copyin_map_to_vars() + self.attach_vars()
        from_list = self.copyout_map_from_vars() + self.detach_vars()
        tofrom_list = self.copy_map_to_from_vars()
        alloc_list = self.create_alloc_vars()
        alloc_list += self.no_create_vars() # TODO not sure about this
        release_list = self.delete_release_vars()
        deviceptr_list = self.deviceptrs()
        present_list = self.present_vars()
        in_arrays_in_body = [
            el for el in arrays_in_body if el not in inout_arrays_in_body
        ]
        already_considered = [
            el.lower() for el in (to_list + from_list + tofrom_list
                                  + alloc_list + deviceptr_list + present_list)
        ]
        if not self.present_by_default():
            to_list += [
                el for el in in_arrays_in_body if el not in already_considered
            ] # note: {val} - {val,other_val} = empty set
            tofrom_list += [
                el for el in inout_arrays_in_body
                if el not in already_considered
            ]
        result = prefix
        if len(to_list):
            result += " map(to:" + ",".join(to_list) + ")"
        if len(from_list):
            result += " map(from:" + ",".join(from_list) + ")"
        if len(tofrom_list):
            result += " map(tofrom:" + ",".join(tofrom_list) + ")"
        if len(alloc_list):
            result += " map(alloc:" + ",".join(alloc_list) + ")"
        if len(release_list):
            result += " map(release:" + ",".join(release_list) + ")"
        if len(deviceptr_list):
            result += " use_device_ptr(" + ",".join(deviceptr_list) + ")"
        # if, async
        if self.async_nowait() != grammar.CLAUSE_NOT_FOUND:
            result += " nowait"
            if len(depend):
                for kind, variables in depend.items():
                    result += " depend(" + kind + ":" + ",".join(
                        variables) + ")"
            else:
                if len(in_arrays_in_body):
                    result += " depend(to:" + ",".join(in_arrays_in_body) + ")"
                if len(inout_arrays_in_body):
                    result += " depend(tofrom:" + ",".join(
                        inout_arrays_in_body) + ")"

        if_condition = self.if_condition()
        if len(if_condition):
            result += " if (" + if_condition + ")"
        return result


class TTAccData(TTAccDataManagementDirectiveBase):
    """
    possible clauses:
      clauses of TTAccDataManagementDirectiveBase
      if( condition )
    """

    def omp_f_str(self, arrays_in_body=[], inout_arrays_in_body=[], depend={}):
        return self._format(
            TTAccDataManagementDirectiveBase.omp_f_str(self, arrays_in_body,
                                                       inout_arrays_in_body,
                                                       depend,
                                                       "!$omp target data"))


class TTAccEnterData(TTAccDataManagementDirectiveBase):
    """
    possible clauses:
      clauses of TTAccData
    """

    def omp_f_str(self, arrays_in_body=[], inout_arrays_in_body=[], depend={}):
        return self._format(
            TTAccDataManagementDirectiveBase.omp_f_str(
                self, arrays_in_body, inout_arrays_in_body, depend,
                "!$omp target enter data"))


class TTAccExitData(TTAccDataManagementDirectiveBase):
    """
    possible clauses:
      clauses of TTAccData
      async [( int-expr )]
      wait [( wait-argument )]
    """

    def omp_f_str(self, arrays_in_body=[], inout_arrays_in_body=[], depend={}):
        return self._format(
            TTAccDataManagementDirectiveBase.omp_f_str(
                self, arrays_in_body, inout_arrays_in_body, depend,
                "!$omp target exit data"))


class TTAccDeclare(TTAccDirectiveBase):
    """ NOTE: In contrast to OMP, '!$acc declare' is only applied to global variables.
    There is '!$acc routine' for routines.
    """

    def map_alloc_vars(self):
        return self.handle_mapping_clause(["create"])

    def map_to_vars(self):
        return self.handle_mapping_clause(["copyin"])

    def map_from_vars(self):
        return self.handle_mapping_clause(["copyout"])

    def map_tofrom_vars(self):
        return self.handle_mapping_clause(["copy"])

    def omp_f_str(self):
        to_list = self.handle_mapping_clause(\
          ["create","copy","copyin","attach","copyout","detach"])

        # TODO: not sure about this mapping
        to_list += self.handle_mapping_clause(["device_resident"])
        link_list = self.handle_mapping_clause(["link"])

        result = "!$omp target declare"
        if len(to_list):
            result += " to(" + ",".join(to_list) + ")"
        if len(link_list):
            result += " link(" + ",".join(link_list) + ")"
        return self._format(result)


class TTAccRoutine(TTAccDirectiveBase):

    def _assign_fields(self, tokens):
        self.id, self.clauses = tokens

    def parallelism(self):
        result = [c for c in self.clauses if c in ["seq", "gang", "worker"]]
        if len(result):
            return result[0]
        else:
            return "seq"

    def omp_f_str(self, additions=""):
        """ additions can be used to pass 'notinbranch' 'inbranch'"""
        result = "!$omp target declare"
        #if base.find_first(self.clauses,TTAccClauseGang) != None:
        #    pass
        #elif base.find_first(self.clauses,TTAccClauseWorker) != None:
        #    pass
        if base.find_first(self.clauses, TTAccClauseVector) != None:
            if self.id != None:
                result += " simd(" + base.make_f_str(self.id) + ")"
            else:
                result += " simd "
        #if not "device_type" in additions:
        #    result += " device_type(any)"
        if len(additions):
            result += " " + additions
        return self._format(result)


class TTAccUpdate(TTAccDirectiveBase):

    def omp_f_str(self):
        result = "!$omp target update"
        host_clause = base.find_first(self.clauses, TTAccClauseHost)
        device_clause = base.find_first(self.clauses, TTAccClauseDevice)
        if host_clause != None:
            result += " to(" + ",".join(host_clause.var_names()) + ")"
        if device_clause != None:
            result += " from(" + ",".join(device_clause.var_names()) + ")"
        return self._format(result)


class TTAccWait(TTAccDirectiveBase):

    def wait_args(self):
        """ Can be used to deduce task dependencies """
        clause = base.find_first(self.clauses, TTAccClauseWait)
        if clause is None:
            return None
        else:
            return clause.expressions()

    def async_queue(self):
        assert False, "Not implemented!"

    def omp_f_str(self, depend_str=""):
        """ :param depend_str: Append depend clauses """
        result = "!$omp taskwait"
        if len(depend_str):
            result += " " + depend_str
        else:
            wait_args = self.wait_args()
            if len(wait_args):
                result += " depend(" + opts.depend_todo + ")"
        return self._format(result)

class AccDeviceSpec(directives.DeviceSpec):
    def __init__(self,device_type,clauses=[]):
        self.device_type = device_type.lower()
        self.clauses = clauses
    def append_clause(self,clause,silently_handle_duplicates=False):
        """:param bool silently_handle_duplicates: Do not throw exception if the clause
                                                exists already, simply do not add it.
        """
        for clause in list(self.clauses):
            if type(new_clause) == type(clause):
                if not silently_handle_duplicates:
                    clause_name = str(type(new_clause).__name__).lower().replace("ttaccclause")
                    raise error.util.SyntaxError("clause '{}' specified more than once '{}' for device type '{}'".format(
                        clause_name,self.device_type))
            else:
                self.clauses.append(clause)

    def applies_to(self,device_type):
        """:return: This device spec applies to the given device_type."""
        if device_type == "*":
            return True
        else:
            return self.device_type == device_type.lower()

    def num_gangs_teams_blocks(self):
        """:note: gang and num_gangs(<arg>) might be specified both on acc parallel/kernels loop construct.
        """
        clause = next((c for c in self.clauses if c.kind in ["gang","num_gangs"]),None)
        return grammar.CLAUSE_NOT_FOUND if clause is None else c.value

    def num_workers(self):
        """:note: worker and num_workers(<arg>) might be specified both on acc parallel/kernels loop construct.
        """
        clause = next((c for c in self.clauses if c.kind in ["worker","num_workers"]),None)
        return grammar.CLAUSE_NOT_FOUND if clause is None else c.value

    def simdlen_vector_length(self):
        """:note: vector and vector_length(<arg>) might be specified both on acc parallel/kernels loop construct.
        """
        clause = next((c for c in self.clauses if c.kind in ["vector","vector_length"]),None)
        return grammar.CLAUSE_NOT_FOUND if clause is None else c.value

    # specific to loop
    def parallelism(self):
        """:return: The parallelism level of this loop.
        :note: Does only make sense for loop device specifications.
        """
        has_gang = next((c for c in self.clauses if c.kind == "gang"),None)
        has_worker = next((c for c in self.clauses if c.kind == "worker"),None)
        has_vector = next((c for c in self.clauses if c.kind == "vector"),None)
        has_seq = next((c for c in self.clauses if c.kind == "seq"),None)
        if has_seq:
            if has_gang or has_worker or has_vector:
                raise util.error.SyntaxError("'seq' clause may not appear with 'gang', 'worker', or 'vector' clause.")
            return directives.ParallelismLevel.SEQ # Most fine-grain loop iterate (gang,worker,vector lane) does run this loop sequentially
        elif has_gang and has_vector: # has_worker can be true or false
            return directives.ParallelismLevel.GANG_VECTOR # loop is mapped to all threads (grids x blocks, all threads active)
        elif has_gang and has_worker:
            return directives.ParallelismLevel.GANG_WORKER # loop is mapped to all warps (1 thread per warp, other threads masked out)
        elif has_worker and has_vector:
            return directives.ParallelismLevel.WORKER_VECTOR # loop is mapped to thread block (all threads per block active)
        elif has_gang:
            return directives.ParallelismLevel.GANG # loop is mapped to grid (1 warp per block active, 1 thread per warp active)
        elif has_worker:
            return directives.ParallelismLevel.WORKER # loop is mapped to warps, number of blocks & warps up to preceding loop (1 thread per warp active) 
        elif has_vector
            return directives.ParallelismLevel.VECTOR # loop is mapped threads in warp, number of blocks & warps up to preceding loops
        else:
            return directives.ParallelismLevel.AUTO # automatically determine parallelism level

    def num_threads_in_block(self):
        workers = self.num_workers()
        vector_len = self.len_simd_vector()
        if workers[0] < 1 or vector_len[0] < 1:
            return [grammar.CLAUSE_NOT_FOUND]
        else:
            return [workers[0] * vector_len[0]]

    def order_of_iterates(self):
        """ Always assume data independent if no seq clause is used """
        seq_clause = next((c for c in self.clauses if c.kind == "seq"),None)
        independent_clause = next(c for c in self.clauses if c.kind == "independent"),None)
        auto_clause = next(c for c in self.clauses if c.kind == "auto"),None)
        if seq_clause != None:
            if auto_clause != None:
                raise error.util.SyntaxError("'auto' clause may not appear together with 'seq' clause")
            return directives.IterateOrder.SEQ
        elif independent_clause != None:
            if auto_clause != None:
                raise error.util.SyntaxError("'auto' clause may not appear together with 'independent' clause")
            return directives.IterateOrder.INDEPENDENT
        else:
            return directives.IterateOrder.AUTO
    
    def num_collapse(self):
        clause = next((c for c in self.clauses if c.kind == "collapse"),None)
        if clause is None:
            return 1
        else:
            return clause.value()

    def tile_sizes(self):
        assert False, "Not implemented!"
        return [1]
    
def __derive_device_specs(acc_clauses,device_specific_clause_kinds):
    """:note:Inserts the '* identifier as last entry into
    the list of device specs to ensure that it is found last
    when comparing device types.
    """
     
    default_spec = AccDeviceSpec(None)
    named_specs = []
    consumed_acc_clauses = []

    current_specs = [default_spec]
    for clause in list(acc_clauses):
        if clause.kind in device_specific_clause_kinds:
            for spec in current_specs:
                current_specs.append_clause(clause)
        elif clause.kind == "device_type":
            current_specs.clear()
            for device_type in clause.device_types:
                if device_type == "*" and len(clause.device_types) > 1:
                    raise util.error.SyntaxError("device type specifier '*' may only appear alone in the argument list of the 'device_type' clause")
                existing_spec = next((spec for spec in named_specs,
                    existing_spec.device_type == device_type.lower()),None)
                if existing_spec != None:
                    raise util.error.SyntaxError("'device_type' clause for device type '{}' may only appear once")
                else:
                    new_spec = AccDeviceSpec(default_spec.clauses)
                    current_specs.append(new_spec)
                    named_specs.append(new_spec)
        elif current_specs[0] != default_spec:
            raise util.error.SyntaxError("only "+",".join(device_specific_clause_kinds)+" clauses may follow a 'device_type' clause")
        consumed_acc_clauses.append(clause)
    # Now single out the '*' spec or make the default spec the "*" spec
    other_spec = AccDeviceSpec("*",default_spec.clauses)
    for spec in list(named_specs):
        if spec.device_type == "*":
            named_specs.remove(spec)
            other_spec = spec
    return (named_specs, other_spec, consumed_acc_clauses)

class TTAccLoop(TTAccDirectiveBase,directives.ILoopAnnotation):
    """
    possible clauses:
      collapse( n )
      gang [( gang-arg-list )]
      worker [( [num:]int-expr )]
      vector [( [length:]int-expr )]
      seq
      auto
      tile( size-expr-list )
      device_type( device-type-list )
      independent
      private( var-list )
      reduction( operator:var-list )
    """
    def _assign_fields(self, tokens):
        TTAccDirectiveBase._assign_fields(self, tokens)
        self.loop_handles_mutual_clauses = True # can be unset by TTAccParallelLoop or TTAccKernelsLoop
        

    def get_device_specs(self):
        device_specific_clause_kinds = ("collapse",
                                        "gang",
                                        "worker",
                                        "vector",
                                        "seq",
                                        "independent",
                                        "auto",
                                        "tile")
        device_specs, other_spec, _ =\
          __derive_device_specs(self.clauses,device_specific_clause_kinds)
        # append '*' to (end of) deviec specs
        return (device_specs, other_spec)

    def reductions(self, converter=base.make_f_str):
        result = {}
        for clause in base.find_all(self.clauses, TTAccClauseReduction):
            contrib = clause.reductions(converter)
            for key, value in contrib.items():
                if not key in result:
                    result[key] = []
                result[key] += value
        return result

    def private_vars(self, converter=base.make_f_str):
        return self.handle_mapping_clause(["private"], converter)

    def omp_f_str(self, loop_type="do", parallel_region="", prefix="!$omp"):
        result = prefix

        if self.loop_handles_mutual_clauses:
            gang_clause = next((c for c in self.clauses,c.kind in ["num_gangs","gang"]),None)
            # TODO use argument
            if gang != None:
                result += " teams distribute"
        if len(parallel_region):
            result += " " + parallel_region
        result += " " + loop_type
        #worker_clause = next((c for c in self.clauses,c.kind in ["num_workers","worker"]),None)
        vector_clause = next((c for c in self.clauses,c.kind in ["vector_length","vector"]),None)
        # TODO use argument
        if vector_clause != None:
             result += " simd"
             if clause.value > 0:
                 result += " simd simdlen(" + str(clause.value) + ")"
        if self.loop_handles_mutual_clauses:
            private_vars = self.private_vars()
            if len(private_vars):
                result += " private(" + ",".join(private_vars) + ")"
        reductions = self.reductions(base.make_f_str)
        if len(reductions):
            for op, values in reductions.items():
                result += " reduction(" + op + ":" + ",".join(values) + ")"
        # collapse
        num_collapse = self.num_collapse()
        if num_collapse > 1:
            result += " collapse({})".format(num_collapse)
        if self.loop_handles_mutual_clauses:
            return self._format(result)
        else:
            return result


class TTAccComputeConstructBase(TTAccDataManagementDirectiveBase,directives.IComputeConstruct):
    """
    possible clauses (for kernels / parallel):
      async [( int-expr )]
      wait [( int-expr-list )]
      num_gangs( int-expr )
      num_workers( int-expr )
      vector_length( int-expr )
      device_type( device-type-list )
      if( condition )
      self [( condition )]
      reduction( operator:var-list )
      copy( var-list )
      copyin( [readonly:]var-list )
      copyout( var-list )
      create( var-list )
      no_create( var-list )
      present( var-list )
      deviceptr( var-list )
      attach( var-list )
      private( var-list )
      firstprivate( var-list )
      default( none | present )
    """
    
    def get_device_specs(self):
        device_specific_clause_kinds = ("num_gangs",
                                        "num_workers",
                                        "vector_length")
        device_specs, other_spec, _ =\
          __derive_device_specs(self.clauses,device_specific_clause_kinds)
        # append '*' to (end of) deviec specs
        return (device_specs, other_spec)
    
    def gang_team_private_vars(self, converter=base.make_f_str):
        return self.handle_mapping_clause(["private"], converter)

    def gang_team_firstprivate_vars(self, converter=base.make_f_str):
        return self.handle_mapping_clause(["firstprivate"], converter)

    def omp_f_str(self,
                  arrays_in_body=[],
                  inout_arrays_in_body=[],
                  depend={},
                  prefix="!$omp target"):
        result = prefix
        data_part = " " + TTAccDataManagementDirectiveBase.omp_f_str(
            self, arrays_in_body, inout_arrays_in_body, depend, "")
        return result + data_part

class TTAccSerial(TTAccComputeConstructBase):

    def is_serial_construct(self):
        """:return: If the outer loops should be mapped
                    to threads. 
        """
        return True

    def omp_f_str(self,
                  arrays_in_body=[],
                  inout_arrays_in_body=[],
                  depend={},
                  prefix="!$omp target"):
        raise util.error.LimitationError("conversion of acc serial construct to OpenMP is not implemented!")


class TTAccParallel(TTAccComputeConstructBase):
   
    def get_device_specs(self):
        device_specific_clause_kinds = ("collapse",
                                        "gang",
                                        "worker",
                                        "vector",
                                        "seq",
                                        "independent",
                                        "auto",
                                        "tile")
        device_specs, other_spec, _ =\
          __derive_device_specs(self.clauses,device_specific_clause_kinds)
        # append '*' to (end of) deviec specs
        return (device_specs, other_spec)

    def omp_f_str(self,
                  arrays_in_body=[],
                  inout_arrays_in_body=[],
                  depend={},
                  prefix="!$omp target"):
        return self._format(
            TTAccComputeConstructBase.omp_f_str(self, arrays_in_body,
                                                inout_arrays_in_body, depend,
                                                prefix))


class TTAccParallelLoop(TTAccLoop,TTAccParallel):
    
    def _assign_fields(self, tokens):
        TTAccParallel._assign_fields(self, tokens)
        self.loop_handles_mutual_clauses = False



    def omp_f_str(self,
                  arrays_in_body=[],
                  inout_arrays_in_body=[],
                  depend={},
                  loop_type="do"):
        """
        :note: loop_type may be changed later if there is a target 'workshare' construct
               in the OpenMP standard. Then, matrix assignments can be mapped
               with such a construct.
        """
        result = "!$omp target teams distribute"
        parallel_part = TTAccParallel.omp_f_str(self, arrays_in_body,
                                                inout_arrays_in_body, depend,
                                                "")
        loop_part = TTAccLoop.omp_f_str(self, loop_type, "parallel", "")
        return self._format(result + " " + loop_part.lstrip() + " "
                            + parallel_part.strip())


class TTAccKernels(TTAccParallel):
    pass


class TTAccKernelsLoop(TTAccParallelLoop):
    pass


# end directives
class TTAccEndData(TTAccDirectiveBase):

    def _assign_fields(self, tokens):
        pass

    def omp_f_str(self):
        return self._format("!$omp end target data")


#
# Connect actions with grammar
#
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
# directive action
grammar.acc_update.setParseAction(TTAccUpdate)
grammar.acc_wait.setParseAction(TTAccWait)
-#acc_host_data #TODO
grammar.acc_data.setParseAction(TTAccData)
grammar.acc_enter_data.setParseAction(TTAccEnterData)
grammar.acc_exit_data.setParseAction(TTAccExitData)
grammar.acc_routine.setParseAction(TTAccRoutine)
grammar.acc_declare.setParseAction(TTAccDeclare)
#acc_atomic #TODO
#acc_cache  #TODO
grammar.acc_loop.setParseAction(TTAccLoop)
# kernels / parallels
#acc_serial #TODO
grammar.acc_serial.setParseAction(TTAccSerial)
grammar.acc_kernels.setParseAction(TTAccKernels)
grammar.acc_parallel.setParseAction(TTAccParallel)
grammar.acc_parallel_loop.setParseAction(TTAccParallelLoop)
grammar.acc_kernels_loop.setParseAction(TTAccKernelsLoop)
grammar.ACC_END_DATA.setParseAction(TTAccEndData)
