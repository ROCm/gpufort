# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import textwrap

import enum

import pyparsing

from gpufort import indexer
from gpufort import util

from .. import prepostprocess
from .. import opts

from . import base
from . import traversals
from . import fortran
from . import grammar
    
class Parallelism(enum.Enum):
    UNSPECIFIED = -2
    AUTO = -1
    SEQ=0
    GANG=1
    WORKER=2
    VECTOR=3
    GANG_WORKER=4
    WORKER_VECTOR=5
    GANG_VECTOR=6
    GANG_WORKER_VECTOR=7

    def gang_partitioned_mode(self):
        return self in [Parallelism.GANG,
                        Parallelism.GANG_WORKER,
                        Parallelism.GANG_WORKER_VECTOR]
    
    def worker_partitioned_mode(self):
        return self in [Parallelism.WORKER,
                        Parallelism.GANG_WORKER,
                        Parallelism.GANG_WORKER_VECTOR]
    
    def vector_partitioned_mode(self):
        return self in [Parallelism.VECTOR,
                        Parallelism.GANG_VECTOR,
                        Parallelism.GANG_WORKER_VECTOR]

class IterateOrder(enum.Enum):
    UNSPECIFIED = -2
    AUTO = -1 # requires static analyis
    SEQ = 0
    INDEPENDENT = 1

class ILoopAnnotation():

    def private_vars(self, converter=traversals.make_fstr):
        """ CUF,ACC: all scalars are private by default """
        return []

    def lastprivate_vars(self, converter=traversals.make_fstr):
        """ only OMP """
        return []

    def reductions(self, converter=traversals.make_fstr):
        """ CUF: Scalar lvalues are reduced by default """
        return {}

    def shared_vars(self, converter=traversals.make_fstr):
        """ only OMP """
        return []

    def all_arrays_are_on_device(self):
        """ only True for CUF kernel do directive """
        return False
    
    def get_device_specs(self):
        return []

class IDeviceSpec():
    def applies_to(self,device_type):
        return True

    def num_collapse(self):
        return grammar.CLAUSE_NOT_FOUND

    def tile_sizes(self):
        return [grammar.CLAUSE_NOT_FOUND]

    def grid_expr_fstr(self):
        """ only CUF """
        return None

    def block_expr_fstr(self):
        """ only CUF """
        return None

    def num_gangs(self):
        return [grammar.CLAUSE_NOT_FOUND]

    def num_threads_in_block(self):
        return [grammar.CLAUSE_NOT_FOUND]

    def num_workers(self):
        """ only ACC """
        return grammar.CLAUSE_NOT_FOUND

    def vector_length(self):
        return grammar.CLAUSE_NOT_FOUND

    def parallelism(self):
        return Parallelism.AUTO

    def order_of_iterates(self):
        return IterateOrder.AUTO 


class TTDo(base.TTContainer):

    def _assign_fields(self, tokens):
        # Assignment, number | variable
        self.annotation, self._begin, self._end, self._step, self.body = tokens
        if self.annotation == None:
            self.annotation = ILoopAnnotation()
        self.thread_index = None # "z","y","x"

    def child_nodes(self):
        return [self.annotation, self.body, self._begin, self._end, self._step]

    def begin_cstr(self):
        return traversals.make_cstr(self._begin._rhs)
    
    def end_cstr(self):
        return traversals.make_cstr(self._end)
   
    def has_step(self):
        return self._step != None

    def step_cstr(self):
        return traversals.make_cstr(self._step)

    # TODO clean up
    def hip_thread_index_cstr(self):
        idx = self.loop_var()
        begin = traversals.make_cstr(
            self._begin._rhs) # array indexing is corrected in index macro
        end = traversals.make_cstr(self._end)
        if self._step != None:
            step = traversals.make_cstr(self._step)
        else:
            step = "1"
        return "int {idx} = {begin} + ({step})*(threadIdx.{tidx} + blockIdx.{tidx} * blockDim.{tidx});\n".format(\
                idx=idx,begin=begin,tidx=self.thread_index,step=step)

    def collapsed_loop_index_cstr(self):
        idx = self.loop_var()
        args = [
          self.thread_index,
          traversals.make_cstr(self._begin._rhs),
          traversals.make_cstr(self._end),
        ]
        if self._step != None:
            args.append(traversals.make_cstr(self._step))
        else:
            args.append("1")
        return "int {idx} = outermost_index({args});\n".format(\
               idx=idx,args=", ".join(args))

    def problem_size(self):
        if self._step == None:
            step = "1"
        else:
            step = traversals.make_fstr(self._step)
        return "gpufort_loop_len(int({begin},c_int),int({end},c_int),int({step},c_int))".format(\
            begin=traversals.make_fstr(self._begin._rhs),end=traversals.make_fstr(self._end),step=step)
    
    def problem_size_cstr(self):
        converter = traversals.make_cstr
        if self._step == None:
            step = "1"
        else:
            step = converter(self._step)
        return "loop_len({begin},{end},{step})".format(\
            begin=converter(self._begin._rhs),end=converter(self._end),step=step)

    # TODO rename
    def hip_thread_bound_cstr(self):
        args = [
          self.loop_var(),
          traversals.make_cstr(self._end),
        ]
        if self._step != None:
            args.append(traversals.make_cstr(self._step))
        else:
            args.append("1")
        return "loop_cond({})".format(",".join(args))

    def loop_var(self, converter=traversals.make_cstr):
        return converter(self._begin._lhs)

    def cstr(self):
        body = textwrap.dedent(base.TTContainer.cstr(self))
        if self.thread_index == None:
            idx = self.loop_var()
            begin = traversals.make_cstr(
                self._begin._rhs) # array indexing is corrected in index macro
            end = traversals.make_cstr(self._end)
            condition = self.hip_thread_bound_cstr()
            step = None
            if self._step != None:
                step = traversals.make_cstr(self._step)
                try: # check if step is an integer number
                    ival = int(step)
                    step = str(ival)
                    if ival > 0:
                        condition = "{} <= {}".format(idx,end)
                    else:
                        condition = "{} >= {}".format(idx,end)
                except:
                    pass
            else:
                condition = "{} <= {}".format(idx,end)
                step = "1"
            return textwrap.dedent("""\
                for ({idx}={begin}; {condition}; {idx} += {step}) {{
                {body}
                }}""").format(\
                    idx=idx, begin=begin, condition=condition, step=step, body=textwrap.indent(body," "*2))
        else:
            return textwrap.indent(body," "*2)


class IComputeConstruct():

    def num_dimensions(self):
        return 1

    def discover_reduction_candidates(self):
        return False

    def get_device_specs(self):
        return []

    def gang_private_vars(self, converter=traversals.make_fstr):
        """ CUF,ACC: all scalars are private by default """
        return []

    def gang_firstprivate_vars(self, converter=traversals.make_fstr):
        return []

    def gang_reductions(self, converter=traversals.make_fstr):
        return {}

    def gang_shared_vars(self, converter=traversals.make_fstr):
        return []

    def local_scalars(self):
        return []

    def reduction_candidates(self):
        return []

    def async_nowait():
        """value != grammar.CLAUSE_NOT_FOUND means True"""
        return grammar.CLAUSE_NOT_FOUND

    def stream(self, converter=traversals.make_fstr):
        return "c_null_ptr"

    def sharedmem(self, converter=traversals.make_fstr):
        return "0"

    def use_default_stream(self):
        return True

    def depend(self):
        """ only OMP """
        #return { "in":[], "out":[], "inout":[], "inout":[], "mutexinoutset":[], "depobj":[] }
        return {}

    def if_condition(self):
        """ OMP,ACC: accelerate only if condition is satisfied. Empty string means condition is satisfied. """
        return ""

    def self_condition(self):
        """ OMP,ACC: run on current CPU / device (and do not offload) """
        return ""

    def deviceptrs(self):
        return []

    def create_alloc_vars(self):
        return []

    def no_create_vars(self):
        """ only ACC"""
        return []

    def present_vars(self):
        """ only ACC"""
        return []

    def delete_release_vars(self):
        return []

    def copy_map_tofrom_vars(self):
        return []

    def copyin_map_to_vars(self):
        return []

    def copyout_map_from_vars(self):
        return []

    def attach_vars(self):
        """ only ACC """
        return []

    def detach_vars(self):
        """ only ACC """
        return []

    def all_mapped_vars(self):
        """:return: Name of all mapped variables. """
        result = []
        result += self.deviceptrs()
        result += self.create_alloc_vars()
        result += self.no_create_vars()
        result += self.present_vars()
        result += self.delete_release_vars()
        result += self.copy_map_tofrom_vars()
        result += self.copyin_map_to_vars()
        result += self.copyout_map_from_vars()
        result += self.attach_vars()
        result += self.detach_vars()
        return result

    def present_by_default(self):
        """ only ACC parallel """
        return True
    
    def is_serial_construct(self):
        return False 

    def cstr(self):
        return ""


class TTComputeConstruct(base.TTContainer):

    def _assign_fields(self, tokens):
        self._parent_directive, self.body = tokens
        self.scope = indexer.types.EMPTY_SCOPE

    def child_nodes(self):
        return [self._parent_directive, self.body]

    def parent_directive(self):
        return self._parent_directive

    def get_device_specs(self):
        return self._parent_directive.get_device_specs()

    def async_nowait():
        """value != grammar.CLAUSE_NOT_FOUND means True"""
        return self.parent_directive().async_nowait()

    def depend(self):
        return self.parent_directive().depend()

    def if_condition(self):
        return self.parent_directive().if_condition()

    def self_condition(self):
        return self.parent_directive().self_condition

    def all_arrays_are_on_device(self):
        return self.parent_directive().all_arrays_are_on_device()

    def create_alloc_vars(self):
        return self.parent_directive().create_alloc_vars()

    def no_create_vars(self):
        return self.parent_directive().no_create_vars()

    def present_vars(self):
        return self.parent_directive().present_vars()

    def delete_release_vars(self):
        return self.parent_directive().delete_release_vars()

    def copy_map_tofrom_vars(self):
        return self.parent_directive().copy_map_tofrom_vars()

    def copyin_map_to_vars(self):
        return self.parent_directive().copyin_map_to_vars()

    def copyout_map_from_vars(self):
        return self.parent_directive().copyout_map_from_vars()

    def attach_vars(self):
        return self.parent_directive().attach_vars()

    def detach_vars(self):
        return self.parent_directive().detach_vars()

    def present_by_default(self):
        return self.parent_directive().present_by_default()

    def is_serial_construct(self):
        return self.parent_directive().is_serial_construct()

    def gang_private_vars(self, converter=traversals.make_fstr):
        result = self.parent_directive().gang_private_vars(converter)
        return result

    def gang_firstprivate_vars(self, converter=traversals.make_fstr):
        return self.parent_directive().gang_firstprivate_vars(converter)

#    # TODO move into analysis
#    def gang_reductions(self, converter=traversals.make_fstr):
#        if self.__first_loop_annotation() != None:
#            if self.__first_loop_annotation().discover_reduction_candidates():
#                return {
#                    "UNKNOWN": self.reduction_candidates()
#                } # TODO default reduction type should be configurable
#            else:
#                return self.__first_loop_annotation().reductions(converter)
#        else:
#            return {}
        

    def stream(self, converter=traversals.make_fstr):
        return self.parent_directive().stream(converter)

    def sharedmem(self, converter=traversals.make_fstr):
        return self.parent_directive().sharedmem(converter)

class TTProcedureBody(base.TTContainer):

    def _assign_fields(self, tokens):
        self.body = tokens
        self.scope = []
        self.result_name = ""

# TODO move to different place
def format_directive(directive_line, max_line_width):
    result = ""
    line = ""
    tokens = directive_line.split(" ")
    sentinel = tokens[0]
    for tk in tokens:
        if len(line + tk) > max_line_width - 1:
            result += line + "&\n"
            line = sentinel + " "
        line += tk + " "
    result += line.rstrip()
    return result
