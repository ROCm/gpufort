# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import textwrap

import pyparsing

from gpufort import indexer
from gpufort import util
from .. import prepostprocess
from .. import opts
from . import base
from . import fortran
from . import grammar
from . import transformations

class ILoopAnnotation():

    def num_collapse(self):
        return grammar.CLAUSE_NOT_FOUND

    def tile_sizes(self):
        return [grammar.CLAUSE_NOT_FOUND]

    def grid_expr_f_str(self):
        """ only CUF """
        return None

    def block_expr_f_str(self):
        """ only CUF """
        return None

    def num_gangs_teams_blocks(self):
        return [grammar.CLAUSE_NOT_FOUND]

    def num_threads_in_block(self):
        return [grammar.CLAUSE_NOT_FOUND]

    def num_workers(self):
        """ only ACC """
        return grammar.CLAUSE_NOT_FOUND

    def simdlen_vector_length(self):
        return grammar.CLAUSE_NOT_FOUND

    def data_independent_iterations(self):
        return True

    def private_vars(self, converter=base.make_f_str):
        """ CUF,ACC: all scalars are private by default """
        return []

    def lastprivate_vars(self, converter=base.make_f_str):
        """ only OMP """
        return []

    def discover_reduction_candidates(self):
        return False

    def reductions(self, converter=base.make_f_str):
        """ CUF: Scalar lvalues are reduced by default """
        return {}

    def shared_vars(self, converter=base.make_f_str):
        """ only OMP """
        return []

    def all_arrays_are_on_device(self):
        """ only True for CUF kernel do directive """
        return False


class TTDo(base.TTContainer):

    def _assign_fields(self, tokens):
        # Assignment, number | variable
        self.annotation, self._begin, self._end, self._step, self.body = tokens
        if self.annotation == None:
            self.annotation = ILoopAnnotation()
        self.thread_index = None # "z","y","x"

    def children(self):
        return [self.annotation, self.body, self._begin, self._end, self._step]

    def hip_thread_index_c_str(self):
        idx = self.loop_var()
        begin = base.make_c_str(
            self._begin._rhs) # array indexing is corrected in index macro
        end = base.make_c_str(self._end)
        if self._step != None:
            step = base.make_c_str(self._step)
        elif opts.all_unspecified_do_loop_step_sizes_are_positive:
            step = "1"
        else:
            step = "({} <= {}) ? 1 : -1".format(begin,end)
        return "int {idx} = {begin} + ({step})*(threadIdx.{tidx} + blockIdx.{tidx} * blockDim.{tidx});\n".format(\
                idx=idx,begin=begin,tidx=self.thread_index,step=step)

    def collapsed_loop_index_c_str(self):
        idx = self.loop_var()
        args = [
          self.thread_index,
          base.make_c_str(self._begin._rhs),
          base.make_c_str(self._end),
        ]
        if self._step != None:
            args.append(base.make_c_str(self._step))
        elif opts.all_unspecified_do_loop_step_sizes_are_positive:
            args.append("1")
        return "int {idx} = outermost_index({args});\n".format(\
               idx=idx,args=",".join(args))

    def problem_size(self, converter=base.make_f_str):
        if self._step == None:
            return "(1 + abs(({end}) - ({begin})))".format(\
                begin=converter(self._begin._rhs),end=converter(self._end),step=converter(self._step) )
        else:
            return "(1 + abs(({end}) - ({begin}))/abs({step}))".format(\
                begin=converter(self._begin._rhs),end=converter(self._end),step=converter(self._step))

    def hip_thread_bound_c_str(self):
        args = [
          self.loop_var(),
          base.make_c_str(self._begin._rhs),
          base.make_c_str(self._end),
        ]
        if self._step != None:
            args.append(base.make_c_str(self._step))
        elif opts.all_unspecified_do_loop_step_sizes_are_positive:
            args.append("1")
        return "loop_cond({})".format(",".join(args))

    def loop_var(self, converter=base.make_c_str):
        return converter(self._begin._lhs)

    def c_str(self):
        body = textwrap.dedent(base.TTContainer.c_str(self))
        if self.thread_index == None:
            idx = self.loop_var()
            begin = base.make_c_str(
                self._begin._rhs) # array indexing is corrected in index macro
            end = base.make_c_str(self._end)
            condition = self.hip_thread_bound_c_str()
            step = None
            if self._step != None:
                step = base.make_c_str(self._step)
                try: # check if step is an integer number
                    ival = int(step)
                    step = str(ival)
                    if ival > 0:
                        condition = "{} <= {}".format(idx,end)
                    else:
                        condition = "{} >= {}".format(idx,end)
                except:
                    pass
            elif opts.all_unspecified_do_loop_step_sizes_are_positive:
                condition = "{} <= {}".format(idx,end)
                step = "1"
            elif not opts.loop_versioning:
                step = "(({} <= {}) ? 1 : -1)".format(begin,end)
            if step != None:
                return textwrap.dedent("""\
                    for ({idx}={begin}; {condition}; {idx} += {step}) {{
                    {body}
                    }}""").format(\
                        idx=idx, begin=begin, condition=condition, step=step, body=textwrap.indent(body," "*2))
            else:
                return textwrap.dedent("""\
                           if ( {begin} <= {end} ) {{
                             for ({idx}={begin}; {idx} <= {end}; {idx}++) {{
                           {body}
                             }}
                           }} else {{
                             for ({idx}={begin}; {idx} >= {end}; {idx}--) {{
                           {body}
                             }}
                           }}""").format(\
                        idx=idx, begin=begin, end=end, step=step, body=textwrap.indent(body," "*4))
        else:
            return textwrap.indent(body," "*2)


class IComputeConstruct():

    def num_collapse(self):
        return grammar.CLAUSE_NOT_FOUND

    def num_dimensions(self):
        return 1

    def discover_reduction_candidates(self):
        return False

    def grid_expr_f_str(self):
        """ only CUF """
        return None

    def block_expr_f_str(self):
        """ only CUF """
        return None
    
    def num_gangs_teams_blocks_specified(self):
        return next((el for el in self.num_gangs_teams_blocks()
                    if el != grammar.CLAUSE_NOT_FOUND),None) != None
    
    def num_threads_in_block_specified(self):
        return next((el for el in self.num_gangs_teams_blocks() 
                    if el != grammar.CLAUSE_NOT_FOUND),None) != None

    def num_gangs_teams_blocks(self):
        return [grammar.CLAUSE_NOT_FOUND]

    def num_threads_in_block(self):
        return [grammar.CLAUSE_NOT_FOUND]

    def gang_team_private_vars(self, converter=base.make_f_str):
        """ CUF,ACC: all scalars are private by default """
        return []

    def gang_team_firstprivate_vars(self, converter=base.make_f_str):
        return []

    def gang_team_reductions(self, converter=base.make_f_str):
        return {}

    def gang_team_shared_vars(self, converter=base.make_f_str):
        return []

    def local_scalars(self):
        return []

    def reduction_candidates(self):
        return []

    def loop_vars(self):
        return []

    def problem_size(self):
        return []

    def async_nowait():
        """value != grammar.CLAUSE_NOT_FOUND means True"""
        return grammar.CLAUSE_NOT_FOUND

    def stream(self, converter=base.make_f_str):
        return "c_null_ptr"

    def sharedmem(self, converter=base.make_f_str):
        return "0"

    def use_default_stream(self):
        return True

    def depend(self):
        """ only OMP """
        #return { "in":[], "out":[], "inout":[], "inout":[], "mutexinoutset":[], "depobj":[] }
        return {}

    def device_types(self):
        return "*"

    def if_condition(self):
        """ OMP,ACC: accelerate only if condition is satisfied. Empty string means condition is satisfied. """
        return ""

    def self_condition(self):
        """ OMP,ACC: run on current CPU / device (and do not offload) """
        return ""

    def deviceptrs(self, scope):
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

    def copy_map_to_from_vars(self):
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
        result += self.copy_map_to_from_vars()
        result += self.copyin_map_to_vars()
        result += self.copyout_map_from_vars()
        result += self.attach_vars()
        result += self.detach_vars()
        return result

    def present_by_default(self):
        """ only ACC parallel """
        return True

    def c_str(self):
        return ""


class TTLoopNest(base.TTContainer, IComputeConstruct):

    def _assign_fields(self, tokens):
        self._parent_directive, self.body = tokens
        self.scope = indexer.scope.EMPTY_SCOPE

    def children(self):
        return [self._parent_directive, self.body]

    def __first_loop_annotation(self):
        return self.body[0].annotation

    def parent_directive(self):
        if self._parent_directive == None:
            return self._first_loop_annotation()
        else:
            return self._parent_directive

    # TODO move into analysis
    def loop_vars(self):
        num_outer_loops_to_map = int(self.parent_directive().num_collapse())
        identifier_names = []
        do_loops = base.find_all(self.body[0], TTDo)
        for loop in do_loops:
            identifier_names.append(loop.loop_var(base.make_f_str))
        if num_outer_loops_to_map > 0:
            return identifier_names[0:num_outer_loops_to_map]
        else:
            return []

    # TODO move into analysis
    def problem_size(self):
        num_outer_loops_to_map = int(self.parent_directive().num_collapse())
        if opts.loop_collapse_strategy == "grid" or num_outer_loops_to_map == 1:
            num_outer_loops_to_map = min(3, num_outer_loops_to_map)
            result = ["-1"] * num_outer_loops_to_map
            do_loops = base.find_all(self.body[0], TTDo)
            for i, loop in enumerate(do_loops):
                if i < num_outer_loops_to_map:
                    result[i] = loop.problem_size()
            return result
        else: # "collapse"
            result = ""
            do_loops = base.find_all(self.body[0], TTDo)
            for loop in reversed(do_loops[0:num_outer_loops_to_map]):
                if len(result):
                    result += "*"
                result += loop.problem_size()
            if len(result):
                return [result]
            else:
                return ["-1"]

    def async_nowait():
        """value != grammar.CLAUSE_NOT_FOUND means True"""
        return self.parent_directive().async_nowait()

    def depend(self):
        return self.parent_directive().depend()

    def device_types(self):
        return self.parent_directive().device_types()

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

    def copy_map_to_from_vars(self):
        return self.parent_directive().copy_map_to_from_vars()

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

    def grid_expr_f_str(self):
        """ only CUF """
        return self.__first_loop_annotation().grid_expr_f_str()

    def block_expr_f_str(self):
        """ only CUF """
        return self.__first_loop_annotation().block_expr_f_str()

    def gang_team_private_vars(self, converter=base.make_f_str):
        result = self.parent_directive().gang_team_private_vars(converter)
        return result

    def gang_team_firstprivate_vars(self, converter=base.make_f_str):
        return self.parent_directive().gang_team_firstprivate_vars(converter)

    # TODO move into analysis
    def gang_team_reductions(self, converter=base.make_f_str):
        if self.__first_loop_annotation().discover_reduction_candidates():
            return {
                "UNKNOWN": self.reduction_candidates()
            } # TODO default reduction type should be configurable
        else:
            return self.__first_loop_annotation().reductions(converter)

    def stream(self, converter=base.make_f_str):
        return self.parent_directive().stream(converter)

    def sharedmem(self, converter=base.make_f_str):
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
