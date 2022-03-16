# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import pyparsing

from gpufort import indexer
from gpufort import util
from .. import prepostprocess
from .. import opts
from . import base
from . import fortran
from . import grammar
from . import transformations

#TODO exclude other annotations as well from this search
#TODO improve
def _search_values_in_subtree(ttnode, search_filter, scope, min_rank=-1):
    def find_all_matching_exclude_directives_(body,
                                              filter_expr=lambda x: True):
        """Find all nodes in tree of type 'searched_type'."""
        result = []

        def traverse_(curr):
            if isinstance(curr, ILoopAnnotation):
                return
            if filter_expr(curr):
                result.append(curr)
            if isinstance(curr,pyparsing.ParseResults) or\
               isinstance(curr,list):
                for el in curr:
                    traverse_(el)
            elif isinstance(curr, base.TTNode):
                for el in curr.children():
                    traverse_(el)

        traverse_(ttnode)
        return result

    tags = []
    for ttvalue in find_all_matching_exclude_directives_(
            ttnode.body,
            search_filter): # includes the identifiers of the function calls
        tag = indexer.scope.create_index_search_tag_for_var(ttvalue.f_str()) # TODO 
        ivar = indexer.scope.search_scope_for_var(scope, tag)
        if ivar["rank"] >= min_rank and\
           not tag in tags: # ordering important
            tags.append(tag)
    return tags

def _vars_in_subtree(ttnode, scope):
    """:return: all identifiers of LValue and RValues in the body."""

    def search_filter(node):
        return isinstance(node,fortran.IValue) and\
               type(node._value) in [fortran.TTDerivedTypeMember,fortran.TTIdentifier,fortran.TTFunctionCallOrTensorAccess]

    result = _search_values_in_subtree(ttnode, search_filter, scope)
    return result


def _arrays_in_subtree(ttnode, scope):

    def search_filter(node):
        return isinstance(node,fortran.IValue) and\
                type(node._value) is fortran.TTFunctionCallOrTensorAccess

    return _search_values_in_subtree(ttnode, search_filter, scope, 1)


def _inout_arrays_in_subtree(ttnode, scope):

    def search_filter(node):
        return type(node) is fortran.TTLValue and\
                type(node._value) is fortran.TTFunctionCallOrTensorAccess

    return _search_values_in_subtree(ttnode, search_filter, scope, 1)


def _flag_tensors(ttcontainer, scope):
    """Clarify types of function calls / tensor access that are not members of a struct."""
    for value in base.find_all(ttcontainer.body, fortran.IValue):
        if type(value._value) is fortran.TTFunctionCallOrTensorAccess:
           try:
              _ = indexer.scope.search_scope_for_var(scope, value.f_str()) # just check if the var exists
              value._value._is_tensor_access = base.True3
           except util.error.LookupError:
              pass

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
        ivar = self.loop_var()
        begin = base.make_c_str(
            self._begin._rhs) # array indexing is corrected in index macro
        step = base.make_c_str(self._step)
        return "int {var} = {begin} + ({step})*(threadIdx.{idx} + blockIdx.{idx} * blockDim.{idx});\n".format(\
                var=ivar,begin=begin,idx=self.thread_index,step=step)

    def collapsed_loop_index_c_str(self):
        idx = self.loop_var()
        begin = base.make_c_str(self._begin._rhs)
        end = base.make_c_str(self._end)
        step = base.make_c_str(self._step)
        return "int {var} = outermost_index({index},{begin},{end},{step});\n".format(\
               var=idx,begin=begin,end=end,step=step,index=self.thread_index)

    def problem_size(self, converter=base.make_f_str):
        if self._step == "1":
            return "(1 + (({end}) - ({begin})))".format(\
                begin=converter(self._begin._rhs),end=converter(self._end),step=converter(self._step) )
        else:
            return "(1 + (({end}) - ({begin}))/({step}))".format(\
                begin=converter(self._begin._rhs),end=converter(self._end),step=converter(self._step))

    def hip_thread_bound_c_str(self):
        ivar = self.loop_var()
        begin = base.make_c_str(self._begin._rhs)
        end = base.make_c_str(self._end)
        step = base.make_c_str(self._step)
        return "loop_cond({0},{1},{2})".format(ivar, end, step)

    def loop_var(self, converter=base.make_c_str):
        return converter(self._begin._lhs)

    def c_str(self):
        body_content = base.TTContainer.c_str(self)
        if self.thread_index == None:
            ivar = self.loop_var()
            begin = base.make_c_str(
                self._begin._rhs) # array indexing is corrected in index macro
            end = base.make_c_str(self._end)
            step = base.make_c_str(self._step)
            return "for ({0}={1}; {0} <= {2}; {0} += {3}) {{\n{4}\n}}".format(\
                    ivar, begin, end, step, body_content)
        else:
            return body_content


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

    def vars_in_body(self):
        return []

    def arrays_in_body(self):
        return []

    def inout_arrays_in_body(self):
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

    def all_unmapped_arrays(self):
        """:return: Name of all unmapped array variables"""
        mapped_vars = self.all_mapped_vars()
        arrays_in_body = self.arrays_in_body()
        return [var for var in arrays_in_body if not var in mapped_vars]

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

    def __parent_directive(self):
        if self._parent_directive == None:
            return self._first_loop_annotation()
        else:
            return self._parent_directive

    def loop_vars(self):
        num_outer_loops_to_map = int(self.__parent_directive().num_collapse())
        identifier_names = []
        do_loops = base.find_all(self.body[0], TTDo)
        for loop in do_loops:
            identifier_names.append(loop.loop_var(base.make_f_str))
        if num_outer_loops_to_map > 0:
            return identifier_names[0:num_outer_loops_to_map]
        else:
            return identifier_names

    def vars_in_body(self):
        return _vars_in_subtree(self, self.scope)

    def arrays_in_body(self):
        return _arrays_in_subtree(self, self.scope)

    def inout_arrays_in_body(self):
        return _inout_arrays_in_subtree(self, self.scope)

    def __local_scalars_and_reduction_candidates(self, scope):
        """
        local variable      - scalar variable that is not read before the assignment (and is no derived type member)
        reduction_candidates - scalar variable that is written but not read anymore 
        NOTE: Always returns Fortran identifiers
        NOTE: The loop variables need to be removed from this result when rendering the corresponding C kernel.
        NOTE: Implementatin assumes that loop condition variables are not written to in loop body. 
        NOTE: When rendering the kernel, it is best to exclude all variables for which an array declaration has been found,
        from the result list. TTCufKernelDo instances do not know of the type of the variables.
        """
        scalars_read_so_far = [
        ] # per line, with name of lhs scalar removed from list
        initialized_scalars = []
        # depth first search
        assignments = base.find_all_matching(
            self.body[0], lambda node: type(node) in [
                fortran.TTAssignment, fortran.TTComplexAssignment, fortran.
                TTMatrixAssignment
            ])
        for assignment in assignments:
            # lhs scalars
            lvalue = assignment._lhs._value
            lvalue_name = lvalue.f_str().lower()
            if type(lvalue) is fortran.TTIdentifier: # could still be a matrix
                definition = indexer.scope.search_scope_for_var(
                    scope, lvalue_name)
                if definition["rank"] == 0 and\
                   not lvalue_name in scalars_read_so_far:
                    initialized_scalars.append(
                        lvalue_name) # read and initialized in
            # rhs scalars
            rhs_identifiers = base.find_all(assignment._rhs,
                                            fortran.TTIdentifier)
            for ttidentifier in rhs_identifiers:
                rvalue_name = ttidentifier.f_str().lower()
                definition = indexer.scope.search_scope_for_var(
                    scope, rvalue_name)
                if definition["rank"] == 0 and\
                   rvalue_name != lvalue_name: # do not include name of rhs if lhs appears in rhs
                    scalars_read_so_far.append(rvalue_name)
        # initialized scalars that are not read (except in same statement) are likely reductions
        # initialized scalars that are read again in other statements are likely local variables
        reduction_candidates = [
            name for name in initialized_scalars
            if name not in scalars_read_so_far
        ]
        local_scalars = [
            name for name in initialized_scalars
            if name not in reduction_candidates
        ] # contains loop variables
        loop_vars = [var.lower() for var in self.loop_vars()]
        for var in list(local_scalars):
            if var.lower() in loop_vars:
                local_scalars.remove(var)
        return local_scalars, reduction_candidates

    def local_scalars(self):
        local_scalars, _ = self.__local_scalars_and_reduction_candidates(
            self.scope)
        return local_scalars

    def reduction_candidates(self):
        _, reduction_candidates = self.__local_scalars_and_reduction_candidates(
            self.scope)
        return reduction_candidates

    def problem_size(self):
        num_outer_loops_to_map = int(self.__parent_directive().num_collapse())
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
        return self.__parent_directive().async_nowait()

    def depend(self):
        return self.__parent_directive().depend()

    def device_types(self):
        return self.__parent_directive().device_types()

    def if_condition(self):
        return self.__parent_directive().if_condition()

    def self_condition(self):
        return self.__parent_directive().self_condition

    def deviceptrs(self):
        if self.__parent_directive().all_arrays_are_on_device():
            return self.arrays_in_body()
        else:
            return self.__parent_directive().deviceptrs()

    def create_alloc_vars(self):
        return self.__parent_directive().create_alloc_vars()

    def no_create_vars(self):
        return self.__parent_directive().no_create_vars()

    def present_vars(self):
        return self.__parent_directive().present_vars()

    def delete_release_vars(self):
        return self.__parent_directive().delete_release_vars()

    def copy_map_to_from_vars(self):
        return self.__parent_directive().copy_map_to_from_vars()

    def copyin_map_to_vars(self):
        return self.__parent_directive().copyin_map_to_vars()

    def copyout_map_from_vars(self):
        return self.__parent_directive().copyout_map_from_vars()

    def attach_vars(self):
        return self.__parent_directive().attach_vars()

    def detach_vars(self):
        return self.__parent_directive().detach_vars()

    def present_by_default(self):
        return self.__parent_directive().present_by_default()

    def grid_expr_f_str(self):
        """ only CUF """
        return self.__first_loop_annotation().grid_expr_f_str()

    def block_expr_f_str(self):
        """ only CUF """
        return self.__first_loop_annotation().block_expr_f_str()

    def gang_team_private_vars(self, converter=base.make_f_str):
        return self.__parent_directive().gang_team_private_vars(converter)

    def gang_team_firstprivate_vars(self, converter=base.make_f_str):
        return self.__parent_directive().gang_team_firstprivate_vars(converter)

    def gang_team_reductions(self, converter=base.make_f_str):
        if self.__first_loop_annotation().discover_reduction_candidates():
            return {
                "UNKNOWN": self.reduction_candidates()
            } # TODO default reduction type should be configurable
        else:
            return self.__first_loop_annotation().reductions(converter)

    def stream(self, converter=base.make_f_str):
        return self.__parent_directive().stream(converter)

    def sharedmem(self, converter=base.make_f_str):
        return self.__parent_directive().sharedmem(converter)

    def omp_f_str(self, f_snippet):
        """
        :note: The string used for parsing was preprocessed. Hence
               we pass the original Fortran snippet here.
        """
        # TODO circular dependency, directives parent module knows about
        # entities in cuf and acc child modules
        from .cuf import TTCufKernelDo
        from .acc import TTAccClauseGang

        # TODO There is only one loop or loop-like expression
        # in a parallel loop.
        # There might me multiple loops or look-like expressions
        # in a kernels region.
        # kernels directives must be split
        # into multiple clauses.
        # In all cases the begin and end directives must
        # be consumed.
        # TODO find out relevant directives
        # TODO transform string
        # TODO preprocess Fortran colon expressions
        inout_arrays_in_body = self.inout_arrays_in_body()
        arrays_in_body = self.arrays_in_body()
        reduction = self.gang_team_reductions()
        depend = self.depend()
        if type(self.__parent_directive()) is TTCufKernelDo:

            def cuf_kernel_do_repl(parse_result):
                nonlocal arrays_in_body
                nonlocal inout_arrays_in_body
                nonlocal reduction
                return parse_result.omp_f_str(arrays_in_body,
                                              inout_arrays_in_body, reduction,
                                              depend), True

            result,_ = util.pyparsing.replace_first(f_snippet,\
                grammar.cuf_kernel_do,\
                cuf_kernel_do_repl)
            return result
        else:

            def acc_compute_repl(parse_result):
                nonlocal arrays_in_body
                nonlocal inout_arrays_in_body
                nonlocal reduction
                return parse_result.omp_f_str(arrays_in_body,
                                              inout_arrays_in_body,
                                              depend), True

            parallel_region = "parallel"

            def acc_loop_repl(parse_result):
                nonlocal arrays_in_body
                nonlocal inout_arrays_in_body
                nonlocal reduction
                nonlocal parallel_region
                result = parse_result.omp_f_str("do", parallel_region)
                parallel_region = ""
                return result, True

            def acc_end_repl(parse_result):
                nonlocal arrays_in_body
                nonlocal inout_arrays_in_body
                nonlocal reduction
                return parse_result.strip() + "!$omp end target", True

            result,_ = util.pyparsing.replace_first(f_snippet,\
                    grammar.acc_parallel | grammar.acc_parallel_loop | grammar.acc_kernels | grammar.acc_kernels_loop,\
                    acc_compute_repl)
            result,_ = util.pyparsing.replace_all(result,\
                    grammar.acc_loop,\
                    acc_loop_repl)
            result,_ = util.pyparsing.replace_first(result,\
                    grammar.Optional(grammar.White(),default="") + ( grammar.ACC_END_PARALLEL | grammar.ACC_END_KERNELS ),
                    acc_end_repl)
            result,_ = util.pyparsing.erase_all(result,\
                    grammar.ACC_END_PARALLEL_LOOP | grammar.ACC_END_KERNELS_LOOP)
            return result

    def c_str(self):
        """
        This routine generates an HIP kernel body.
        """
        # 0. Clarify types of function calls / tensor access that are not
        # members of a struct
        _flag_tensors(self, self.scope)
        # TODO look up correct signature of called device functions from index
        # 1.1 Collapsing
        num_outer_loops_to_map = int(self.__parent_directive().num_collapse())
        if opts.loop_collapse_strategy == "grid" and num_outer_loops_to_map <= 3:
            dim = num_outer_loops_to_map
        else: # "collapse" or num_outer_loops_to_map > 3
            dim = 1
        tidx = "__gidx{dim}".format(dim=dim)
        # 1. unpack colon (":") expressions
        for expr in base.find_all(self.body[0], fortran.TTStatement):
            if type(expr._statement[0]) is fortran.TTAssignment:
                expr._statement[0] = expr._statement[
                    0].convert_to_do_loop_nest_if_necessary()
        # 2. Identify reduced variables
        for expr in base.find_all(self.body[0], fortran.TTAssignment):
            for value in base.find_all_matching(
                    expr, lambda x: isinstance(x, fortran.IValue)):
                if type(value._value) in [
                        fortran.TTDerivedTypeMember, fortran.TTIdentifier
                ]:
                    for op, reduced_vars in self.gang_team_reductions(
                    ).items():
                        if value.name().lower() in [
                                el.lower() for el in reduced_vars
                        ]:
                            value._reduction_index = tidx
            # TODO identify what operation is performed on the highest level to
            # identify reduction op
        reduction_preamble = ""
        # 2.1. Add init preamble for reduced variables
        for kind, reduced_vars in self.gang_team_reductions(
                base.make_c_str).items():
            for var in reduced_vars:
                if opts.fortran_style_tensor_access:
                    reduction_preamble += "reduce_op_{kind}::init({var}({tidx}));\n".format(
                        kind=kind, var=var, tidx=tidx)
                else:
                    reduction_preamble += "reduce_op_{kind}::init({var}[{tidx}]);\n".format(
                        kind=kind, var=var, tidx=tidx)
        # 3. collapse and transform do-loops
        do_loops = base.find_all(self.body[0], TTDo)
        if num_outer_loops_to_map == 1 or (opts.loop_collapse_strategy
                                           == "grid" and
                                           num_outer_loops_to_map <= 3):
            indices, conditions = transformations.map_loopnest_to_grid(do_loops,num_outer_loops_to_map)
        else: # "collapse" or num_outer_loops_to_map > 3
            indices, conditions = transformations.collapse_loopnest(do_loops)
        
        c_snippet = "{0}\n{2}if ({1}) {{\n{3}\n}}".format(\
            "".join(indices),"&&".join(conditions),reduction_preamble,base.make_c_str(self.body[0]))
        return prepostprocess.postprocess_c_snippet(c_snippet)


class TTProcedureBody(base.TTContainer):

    def _assign_fields(self, tokens):
        self.body = tokens
        self.scope = []
        self.result_name = ""

    def vars_in_body(self):
        """:return: all identifiers of LValue and RValues in the body.
        """
        return _vars_in_subtree(self, self.scope)

    def arrays_in_body(self):
        return _arrays_in_subtree(self, self.scope)

    def inout_arrays_in_body(self):
        return _inout_arrays_in_subtree(self, self.scope)

    def c_str(self):
        """
        :return: body of a procedure as C/C++ code.
        Non-empty result names will be propagated to
        all return statements.
        """
        # 0. Clarify types of function calls / tensor access that are not
        # members of a struct
        _flag_tensors(self, self.scope)
        # 2. Propagate result variable name to return statements
        if len(self.result_name):
            for expr in base.find_all(self.body, TTReturn):
                expr._result_name = result_name
        c_body = base.make_c_str(self.body)

        if len(self.result_name):
            c_body += "\nreturn " + result_name + ";"
        return prepostprocess.postprocess_c_snippet(c_body)


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
