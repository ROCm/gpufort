# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import textwrap

from gpufort import util

from .. import opts
from .. import analysis
from .. import tree
from .. import optvals

from . import loops
from . import loopmgr
from . import resulttypes
from . import arrayexpr

#class HipKernelArgument:
#   def __init__(
#      self,
#      symbol_info,
#      fstr,
#      cstr,
#      rank
#    ):
#      self._symbol_info = symbol_info
#      self._fstr = fstr
#      self._rank = rank
#   @property
#   def fortran_arg_as_str(self):
#       pass
#   @property
#   def kernel_launcher_arg_as_str(self):
#       pass
#   @property
#   def kernel_arg_as_str(self):
#       pass

class MaskedCode:
    """This class associates C code with a mask that enables it.
    """

    def __init__(self,code,mask,indent=""):
        """
        :param str code: C/C++ code.
        :param str mask: condition that enables the code, or None. None is treated as `true`.
        """
        self.mask = mask
        self.code = code
        self.indent = indent
    def mask_matches(self,other):
        if self.mask == other.mask:
            return True
        elif not self.has_mask and not other.has_mask:
            return True
        return False
    @property
    def has_mask(self):
        return self.mask not in [None,"true",""]
    def str(self):
        return textwrap.indent(self.code,self.indent)

class MaskedDummyStatement(tree.TTDummy):
    pass

class __HIPKernelBodyGenerator:

    """
    Used:

    * loops.Loopnest: Generates C++ code from a list of annotated loops.
    * loopmgr.LoopnestManager
    
    Optimizations:
    
    * [unimplemented] Reorder statements such that a more aggressive parallelization can be applied.
                      If a statement is found within a loop body that breaks the tight nesting
                      of a loopnest, move this statement into the body of the next loop (and so on),
                      till the desired number of loops can be collapsed.
    * 
    """

    def __init__(self):
        # options
        self.single_level_indent = "" 
        self.map_to_flat_arrays = False
        self.map_to_flat_scalars = False
        self.reorder_statements_to_maximize_parallelism = False
        # traversal state:
        self._masked_code_list = []
        self._indent = ""
        self._scope = None
        self._compute_construct_info = None
        self._resource_filter = None
        self._loopnest_mgr = None
        #
        self._result = None
        self._optional_args = None
   
    # Statement filtering
    
    def _add_masked_code(self,code):
        self._masked_code_list.append(
          MaskedCode(
            code,
            self._resource_filter.statement_selection_condition(),
            self._indent
          )
        )
    
    def _add_unmasked_code(self,code):
        self._masked_code_list.append(
          MaskedCode(code,None,self._indent)
        )

    def _render_mask_open(self,mask,indent=""):
        #return textwrap.indent(
        #  "if ( {} ) {{\n".format(mask),
        #  indent
        #)
        return textwrap.indent(
          "GPUFORT_MASK_SET ( {} )\n".format(mask),
          indent
        )
    
    def _render_mask_close(self,mask,indent=""):
        #return textwrap.indent(
        #  "}} // {}\n".format(mask),
        #  indent
        #)
        return textwrap.indent(
          "GPUFORT_MASK_UNSET ( {} )\n".format(mask),
          indent
        )
    
    def _render_masked_code_list(self):
        """:note:Detects contiguous blocks of statements with the same mask."""
        result = "" 
        last = MaskedCode(None,None)
        for masked_code in self._masked_code_list:   
            if not masked_code.mask_matches(last): 
                if last.has_mask:
                   result += self._render_mask_close(last.mask)
                if masked_code.has_mask:
                   result += self._render_mask_open(masked_code.mask)
            result += masked_code.str().rstrip("\n") + "\n"
            last = masked_code
        if masked_code.has_mask:
            result += self._render_mask_close(masked_code.mask)
        return result
    # Traversal

    def _check_loop_parallelism(self,
        resource_filter,
        acc_loop_info
      ):
        if acc_loop_info.gang.specified:
            if resource_filter.gang_partitioned_mode():
                raise util.eror.SyntaxError("already in gang-partitioned region")
            elif resource_filter.worker_partitioned_mode():
                raise util.eror.SyntaxError("no gang partitioning possible in worker-partitioned region")
            elif resource_filter.vector_partitioned_mode():
                raise util.eror.SyntaxError("no gang partitioning possible in vector-partitioned region")
        if acc_loop_info.worker.specified:
            if resource_filter.worker_partitioned_mode():
                raise util.eror.SyntaxError("already in worker-partitioned region")
            elif resource_filter.vector_partitioned_mode():
                raise util.eror.SyntaxError("no worker partitioning possible in vector-partitioned region")
        if acc_loop_info.vector.specified:
            if resource_filter.vector_partitioned_mode():
                raise util.eror.SyntaxError("already in vector-partitioned region")
            
    def _traverse_container_body(self,ttcontainer,indent=None):
        """Traverses a container's children and 
        applies the resource filter's statement
        selection filter to contiguous groups
        of non-container statements.
        """
        previous_indent = self._indent
        if indent != None:
            self._indent += indent
        else:
            self._indent += ttcontainer.indent
        for child in ttcontainer:
            self._traverse(child)
        self._indent = previous_indent

    def _traverse_acc_compute_construct(self,ttnode):
        acc_construct_info = analysis.acc.analyze_directive(ttnode,self._result.device_type)  
        if acc_construct_info.is_serial:
            self._result.max_num_gangs = "1"    
            self._result.max_num_workers = "1"    
            self._result.max_vector_length = "1"    
        else:
            if acc_construct_info.num_gangs.specified:
                self._result.max_num_gangs = acc_construct_info.num_gangs
            if acc_construct_info.num_workers.specified:
                self._result.max_num_workers = acc_construct_info.num_workers
            if acc_construct_info.vector_length.specified:
                self._result.max_vector_length = acc_construct_info.vector_length
        if acc_construct_info.private_vars.specified: 
            self._result.private_vars = acc_construct_info.private_vars
            self._result.generated_code += render.render_private_vars_decl_list(
              ttvalues,scope
            )
        self._result.generated_code += loops.render_hip_kernel_prolog_acc()
        if acc_construct_info.firstprivate_vars.specified: 
            self._result.firstprivate_vars = acc_construct_info.firstprivate_vars
        self._compute_construct_info = acc_construct_info

    def _traverse_values(self,values):
        # overwrite: c_name for derived type members if appropriate
        for ttvalue in values:
            if ttvalue.is_derived_type_member_expr:
                if ttvalue.is_scalar and self.map_to_flat_scalars:
                    cname = util.parsing.mangle_fortran_var_expr(ttvalue.cname)
                    ttvalue.overwrite_c_repr(cname,[])
                elif ttvalue.yields_array and self.map_to_flat_arrays:
                    cname = util.parsing.mangle_fortran_var_expr(ttvalue.cname)
                    type_defining_node = ttvalue.type_defining_node
                    rank_defining_node = ttvalue.rank_defining_node
                    if type_defining_node != rank_defining_node:
                        raise util.error.TransformationError("cannot flatten array")
                    if isinstance(rank_defining_node,tree.TTFunctionCall):
                        ttvalue.overwrite_c_repr(cname,list(rank_defining_node.args))
                    else:
                        ttvalue.overwrite_c_repr(cname,[])
            elif ttvalue.is_function_call_expr:
                ttfunccall = ttvalue.value
                if (ttfunccall.is_function_call
                   and ttfunccall.is_intrinsic_call):
                     func_name = ttfunccall.name.lower() 
                     if func_name == "present":
                         # add optional argument
                         pass
                     elif ttfunccall.is_elemental_call:
                        ttfunccall.overwrite_cstr(
                          "_"+func_name,
                          list(ttfunccall.rank_defining_node.args)
                        )

    def _find_rvalues_in_directive(self,ttnode):
        rvalues = []
        analysis.acc.find_rvalues_in_directive(
          ttnode,
          rvalues
        )
        self._result.rvalues += rvalues
        self._traverse_values(rvalues)
    
    def _find_lvalues_rvalues_in_arith_expr(self,ttnode):
        lvalues = []
        rvalues = []
        analysis.fortran.find_lvalues_and_rvalues(
          ttnode,
          lvalues,
          rvalues
        )
        self._result.lvalues += lvalues
        self._result.rvalues += rvalues
        self._traverse_values(lvalues)
        self._traverse_values(rvalues)
   
    def _traverse_acc_loop_directive(self,ttnode):  
        """Create new AccLoopnestManager instance. Append it to the result's list.
        Render it if no collapse or tile clause is specified.
        Search certain clauses for rvalues and lvalues.
        """
        #
        acc_loop_info = analysis.acc.analyze_directive(
          ttnode,
          self._result.device_type
        ) 
        # loop directives might contain rvalues that need to be passed
        self._find_rvalues_in_directive(ttnode)
        self._check_loop_parallelism(
          self._resource_filter,
          acc_loop_info
        )
        #
        self._loopnest_mgr = loopmgr.AccLoopnestManager(acc_loop_info)
    
    def _traverse_cuf_kernel_do_construct(self,ttnode):
        cuf_construct_info = analysis.cuf.analyze_directive(ttnode)  
        if cuf_construct_info.grid.specified:
            self._result.grid = cuf_construct_info.grid
        if cuf_construct_info.block.specified:
            self._result.block = cuf_construct_info.block
        if cuf_construct_info.sharedmem.specified:
            self._result.sharedmem = cuf_construct_info.sharedmem
        if cuf_construct_info.stream.specified:
            self._result.stream = cuf_construct_info.stream
        if cuf_construct_info.reduction.specified: 
            self._result.reductions = cuf_construct_info.reduction
    
    def _traverse_cuf_kernel_do_loop_directive(self,ttdo):  
        """Create new AccLoopnestManager instance. Append it to the result's list.
        Render it if no number of loops is specified. 
        Search certain clauses for rvalues and lvalues.
        """
        #todo: split annotation from loop, init AccLoopnestManager solely with acc_loop_info
        cuf_loop_info = analysis.cuf.analyze_directive(
          ttdo.annotation
        ) 
        acc_loop_info = analysis.acc.AccLoopInfo(None)
        acc_loop_info.gang.specified = True
        acc_loop_info.worker.specified = True
        acc_loop_info.vector.specified = True
        #
        if cuf_loop_info.num_loops.value.specified:
            acc_loop_info.collapse.value = cuf_loop_info.num_loops.value
        self._init_loopnest_mgr(ttdo,acc_loop_info)

    def _traverse_container(self,ttnode):
        """:note: Container statements are never subject to masks."""
        self._find_lvalues_rvalues_in_arith_expr(ttnode)
        self._result.generated_code += textwrap.indent(
          ttnode.header_cstr(),
          self._indent
        )
        previous_indent = self._indent
        self._indent += ttnode.indent
        #
        self._traverse_container_body(ttnode)
        #
        self._indent = previous_indent
        self._result.generated_code += textwrap.indent(
          ttnode.footer_cstr(),
          self._indent
        )

    def _unpack_render_result_and_descend(self,ttdo,render_result):
        loopnest_open,loopnest_close,\
        loopnest_resource_filter,loopnest_indent =\
          render_result
        self._add_unmasked_code(loopnest_open)

        self._resource_filter += loopnest_resource_filter
        previous_indent = self._indent
        self._indent += loopnest_indent
        #
        self._traverse_container_body(ttdo,indent="")
        # 
        self._resource_filter -= loopnest_resource_filter
        self._indent = previous_indent
        self._add_unmasked_code(loopnest_close)
 
    def _render_loopnest_and_descend(self,ttdo_last,save_loopnest_mgr=True):
        """Renders a loopnest, appends it to the result, add the generated
        code to the result's variable, descend into the body
        of the associated container""" 
        render_result = self._loopnest_mgr.map_loopnest_to_hip_cpp(self._scope)
        if save_loopnest_mgr:
            self._result.loopnest_mgr_list.append(self._loopnest_mgr)
        self._loopnest_mgr = None
        # 
        self._unpack_render_result_and_descend(ttdo_last,render_result)
 
    def _traverse_do_loop(self,ttdo):
        # todo: when collapsing, check if we can move statement into inner loop, should be possible if 
        # a loop statement is not relying on the lvalue or inout arg from a previous statement (difficult to analyze)
        # alternative interpretation of collapse user information -> we can reorder statements without error
        # self.reorder_statements_to_maximize_parallelism
        if self._loopnest_mgr != None:
            if not self._loopnest_mgr.iscomplete():
                self._loopnest_mgr.append_do_loop(ttdo)
                if self._loopnest_mgr.iscomplete():
                    self._render_loopnest_and_descend(ttdo)
                else:
                    self._traverse_container_body(ttdo,"")
        else:
            render_result = loopmgr.create_simple_loop(
              ttdo).map_to_hip_cpp(self._scope)
            #:todo: fix statement filter
            self._unpack_render_result_and_descend(ttdo,render_result)

    def _render_linear_idx_offset(self,
        cname,
        indices,
        is_full_array
      ):
        if is_full_array:
            return "0"
        else:
            return "{cname}.c_idx({args})".format(
              cname = cname,
              args = ",".join(indices)
            )

    def _generate_loopnest_for_contiguous_array(self,ttvalue,yields_full_array):
        """
        :return: A list with a single entry: 
                 A dummy TTDo that can be used to generate C++ code
                 and C++/Fortran problem size information.
        :note: The procedure assumes that the Fortran representation of the do loop
               will only be used to calcuate the index range in Fortran code. 
        :note: Contiguous arrays may not contain index ranges with stride > 1.
        :note: Assumes that the index ranges come first,
               as guaranteed by the semantics check.
        :note: Assumes that only the first index range may have
               a specified lbound and only 
               the last index range may have an upper bound.
        :note: Assumes that the array is represented as C++ type
               with a function `c_idx(int i1, ..., int i_rank)` 
               and `lbound(dim)`. 
        :note: Uses the cname member of the TTValue object as the expressions original
               name may have been overriden with a particular name.
               This would, e.g., be the case if members of a derived type are passed separately to a kernel
               instead of the full derived type.
        """
        cname = ttvalue.cname
        if yields_full_array:
            len_cstr = "{}.size()".format(cname)
        else:
            lbounds, ubounds, steps = ttvalue.loop_bounds_as_str(
              only_consider_index_ranges=False,
              converter = tree.traversals.make_cstr
            ) 
            first_idx = self._render_linear_idx_offset(
              cname,
              lbounds,
              yields_full_array
            )
            last_idx  = self._render_linear_idx_offset(
              cname,
              ubounds,
              yields_full_array
            )
            len_cstr = last_idx + " - " + first_idx
        # create dummy do loop
        first = MaskedDummyStatement(
          fstr = "1",
          cstr = "1"
        )
        last = MaskedDummyStatement(
          fstr = "size({})".format(ttvalue.fstr()),
          cstr = len_cstr
        )
        index = tree.TTIdentifier([loops.unique_label("idx")])
        begin = tree.TTAssignment([index,first])
        step = None
        body = []
        return [
          tree.TTDo([begin,last,step,body])
        ]
    
    def _generate_loopnest_for_generic_subarray(self,ttvalue):
        """Loopnest generator generic subarray expressions.
        :return: A nest of dummy TTDos that can be used to generate C++ code
                 Fortran problem size information. 
        """
        loops = []
        lbounds_as_cstr, ubounds_as_cstr, steps_as_cstr = ttvalue.loop_bounds_as_str(
          True,
          tree.traversals.make_cstr
        ) 
        lbounds_as_fstr, ubounds_as_fstr, steps_as_fstr = ttvalue.loop_bounds_as_str(
          True,
          tree.traversals.make_fstr
        ) 
        # create dummy do-loop nest, go from inner to outer loop, left to right
        
        for (lbound_as_fstr,ubound_as_fstr,step_as_fstr,
             lbound_as_cstr,ubound_as_cstr,step_as_cstr) in zip(
                lbounds_as_fstr,ubounds_as_fstr,steps_as_fstr,
                lbounds_as_cstr,ubounds_as_cstr,steps_as_cstr):
            first = MaskedDummyStatement(
              cstr = lbound_as_cstr,
              fstr = lbound_as_fstr
            )
            last = MaskedDummyStatement(
              cstr = ubound_as_cstr,
              fstr = ubound_as_fstr
            )
            index = tree.TTIdentifier([loops.unique_label("idx")])
            begin = tree.TTAssignment([index,first])
            if step_as_fstr != None:
                step = MaskedDummyStatement
            body = []
            loops.append(tree.TTDo([begin,last,step,body]))
        return loops

    def _nest_list_of_loops(self,loops,innermost_loop_body):
        """Nests a linear list of loops so that the do-loop traversal algorithm can be used.
        :param list loops: A list of loops.
        :param list innermost_loop_body: A list of statement nodes to put into
                                           the body of the innermost loop.
        :return: Reference to the first loop.
        """
        assert len(loops)
        first = loops[0]
        last = loops[0]
        for loop in loops[1:]:
            last.body.append(loop)
            last = loop
        last.body += innermost_loop_body
        return first  
    
    def _modify_c_repr_of_contiguous_array_assignment(self,ttassignment,index_arg_cstr):
        assert (ttassignment.rhs.yields_contiguous_array
               or ttassignment.rhs.yields_scalar)
        assert not ttassignment.rhs.applies_transformational_functions_to_arrays
        assert not ttassignment.rhs.contains_array_access_with_index_array
        for ttnode in ttassignment.walk_preorder():
            if isinstance(ttnode,tree.TTValue):
                if ttnode.is_contiguous_array:
                    lbounds, ubounds, steps = ttnode.loop_bounds_as_str(
                      False,
                      tree.traversals.make_cstr
                    ) 
                    if not ttnode.is_full_array:
                        rhs_offset_cstr  = self._render_linear_idx_offset(
                          ttnode.cname,
                          lbounds,
                          ttnode.is_full_array
                        )
                        index_arg_cstr += " + " + rhs_offset_cstr
                    ttnode.overwrite_c_repr(
                      ttnode.cname,
                      [index_arg_cstr],
                      True # square brackets
                    )

    def _create_default_loopnest_mgr_for_array_operation(self,num_collapse=1):
        """default parallelism for mapping array operations.
           Per default, nothing is specified
        """
        if isinstance(self._compute_construct_info,analysis.acc.AccConstructInfo):
            if self._compute_construct_info.is_kernels:
                acc_loop_info = analysis.acc.AccLoopInfo(self._result.device_type)
                acc_loop_info.gang.specified = True
                acc_loop_info.vector.specified = True
                acc_loop_info.vector.num_collapse = 1
                self._loopnest_mgr = loopmgr.AccLoopnestManager(acc_loop_info)

    def _traverse_array_assignment(self,ttassignment):
        """
        - generate collapsed loopnest if one of the expressions is not 
          contiguous or full array
         - assign collapsed loop index directly to full_array 
          - assign collapsed loop index + offset directly to contiguous array
        - generate 
        :note: Currently, contiguous arrays
        """
        # TODO transform before entering
        # TODO transformation is 
        # need to construct loop/loopnest, need to know collapsed index
        assert not ttassignment.rhs.applies_transformational_functions_to_arrays
        if ttassignment.lhs.is_contiguous_array:
            use_single_loop_and_offsets = (
              ttassignment.rhs.yields_contiguous_array
              or ttassignment.rhs.yields_scalar
            )
            if use_single_loop_and_offsets:
                loops = self._generate_loopnest_for_contiguous_array(
                  ttassignment.lhs,
                  ttassignment.lhs.is_full_array
                )
                self._modify_c_repr_of_contiguous_array_assignment(
                  ttassignment, 
                  loops[0].index.cstr()
                )
                dummy = MaskedDummyStatement(
                  cstr = ttassignment.cstr(),
                  fstr = ttassignment.fstr()
                )
                # reduction?
                self._create_default_loopnest_mgr_for_array_operation(len(loops))
                first_loop = self._nest_list_of_loops(loops,[dummy]) # there is only one
                # define and pass down acc information
                self._traverse_do_loop(first_loop)
        else:
            raise util.error.LimitationError("Not implemented yet!")
        #    else:
        #        # work with collapsed loopnest
        #        # assign collapsed index as carg to lhs
        #        assert ttassignment.rhs.yields_array
        #        loops = self._generate_loopnest_for_generic_subarray(
        #          ttassignment.lhs
        #        )
        #        acc_loop_info.collapse = len(loops)
        #        if ttassignment.lhs.yields_full_array:
        #            pass
        #        else:
        #            pass
        #        pass
        #else: # ttassignment is generic subarray
        #    if (ttassignment.rhs.yields_contiguous_array
        #       or ttassignment.rhs.yields_scalar)
        #        # work with collapsed loopnest
        #        # assign collapsed index as bracket carg to rhs
        #        assert ttassignment.rhs.yields_array
        #        loops = self._generate_loopnest_for_generic_subarray(
        #          ttassignment.lhs
        #        )
        #        acc_loop_info.collapse = len(loops)
        #        pass
        #    elif ttassignment.rhs.yields_array:
        #        # collapsed loopnest
        #        # assign un-collapsed indices to rhs and lhs, no carg
        #        assert ttassignment.rhs.yields_array
        #        loops = self._generate_loopnest_for_generic_subarray(
        #          ttassignment.lhs
        #        )
        #        acc_loop_info.collapse = len(loops)

    def _traverse(self,ttnode):
        if isinstance(ttnode,(tree.TTAccParallelLoop,tree.TTAccKernelsLoop)):
            self._traverse_acc_compute_construct(ttnode)
            self._traverse_acc_loop_directive(ttnode)
            self._traverse_container_body(ttnode,indent="")
        elif isinstance(ttnode,(tree.TTAccParallel,tree.TTAccKernels)):
            if (isinstance(ttnode,tree.TTAccKernels) 
               and ttnode.is_device_to_device_copy):
                pass
            self._traverse_acc_compute_construct(ttnode)
            self._traverse_container_body(ttnode,indent="")
        elif isinstance(ttnode,tree.TTCufKernelDo):
            self._traverse_cuf_kernel_do_construct(ttnode)
            self._traverse_container_body(ttnode,indent="")
        elif isinstance(ttnode,tree.TTAccLoop):
            if self._loopnest_mgr == None:
                self._traverse_acc_loop_directive(ttnode)
            self._traverse_container_body(ttnode,indent="")
        elif isinstance(ttnode,tree.TTDo):
            self._traverse_do_loop(ttnode)
        elif isinstance(ttnode,tree.TTContainer):
            self._traverse_container(ttnode)
        elif isinstance(ttnode,tree.TTAssignment):
            if ttnode.lhs.is_array:
                if ttnode.rhs.applies_transformational_functions_to_arrays:
                    raise util.error.LimitationError(
                      "found transformational function in array assignment expression"
                    )
                else:
                    self._traverse_array_assignment(ttnode)
            else: # result is a scalar
                if ttnode.rhs.applies_transformational_functions_to_arrays:
                    assert ttnode.is_scalar 
                    raise util.error.LimitationError("found reduction operation")
                    pass # check for reductions, otherwise fail
                else:
                    self._add_masked_code(ttnode.cstr())
        elif isinstance(ttnode,MaskedDummyStatement):
            self._add_masked_code(ttnode.cstr())
        elif isinstance(ttnode,tree.TTSubroutineCall):
            # must be masked in OpenACC context, is always masked in CUF context, see rule on acc routine clauses,
            # * acc routine worker may not be called from worker and vector but only from gang-part./red. region
            # * acc routine vector may not be called from vector but only from gang-part./red. and worker-part. region
            # * acc routine seq may only be called from gang-part/-red. and worker-/vector-part. regions
            self._add_masked_code(ttnode.cstr())
        else:
            self._add_unmasked_code(ttnode.cstr())

    def map_to_hip_cpp(self,
        ttcomputeconstruct,
        scope,
        device_type
      ):
        """Transform an OpenACC compute construct or routine body to HIP C++.
        :param scope: A scope object, see GPUFORT's indexer component.
        :param str device_type: The device type (`nvidia`, `radeon` or None).
        :param bool initially_gang_partitioned: Start in gang-partitioned mode.
        :param bool initially_worker_partitioned: Start in worker-partitioned mode.
        :param bool initially_vector_partitioned: Start in vector-partitioned mode.
        """
        loops.reset() # reset variable counters of loops package
        self._scope = scope
        self._result = resulttypes.TransformationResult(device_type)
        self._resource_filter = loops.AccResourceFilter()
        self._indent = ""
        self._loopnest_mgr = None
        self._traverse(ttcomputeconstruct)
        self._result.generated_code = self._render_masked_code_list()
        # add the prolog
        return self._result

__instance = __HIPKernelBodyGenerator()

def map_to_hip_cpp(
    ttcomputeconstruct,
    scope,
    device_type = None
  ):
    """Transform an OpenACC compute construct or routine body to HIP C++.
    :param scope: A scope object, see GPUFORT's indexer component.
    :param str device_type: The device type (`nvidia`, `radeon` or None).
     """
    loops.single_level_indent = opts.single_level_indent
    __instance.single_level_indent = opts.single_level_indent
    __instance.map_to_flat_arrays = opts.map_to_flat_arrays
    __instance.map_to_flat_scalars = opts.map_to_flat_scalars
    return __instance.map_to_hip_cpp(
      ttcomputeconstruct,
      scope,
      device_type
    )
