# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
"""
"""

import textwrap

from gpufort import util

from .. import opts
from .. import analysis
from .. import tree
from .. import optvals

from . import loops
from . import assignments
from . import loopmgr
from . import resulttypes
from . import render

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

class HIPKernelBodyGenerator:

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
        self._masked_code_list = render.MaskedCodeList()
        self._indent = ""
        self._scope = None
        self._compute_construct = None
        self._resource_filter = None
        self._loopnest_mgr = None
        #
        self._result = None
        self._optional_args = None
   
    # Statement filtering
    def _add_masked_code(self,code):
        self._masked_code_list.add_masked_code(
          code, 
          self._resource_filter.statement_selection_condition(),
          self._indent
        )
    
    def _add_unmasked_code(self,code):
        self._masked_code_list.add_unmasked_code(
          code,
          self._indent
        )

    # Traversal

    def _check_loop_parallelism(self,
        resource_filter,
        ttaccdir
      ):
        for clause in ttaccdir.walk_clauses_device_type(
              self._result.device_type
          ):
            if isinstance(clause,tree.TTAccClauseGang):
                if resource_filter.gang_partitioned_mode():
                    raise util.eror.SyntaxError("already in gang-partitioned region")
                elif resource_filter.worker_partitioned_mode():
                    raise util.eror.SyntaxError("no gang partitioning possible in worker-partitioned region")
                elif resource_filter.vector_partitioned_mode():
                    raise util.eror.SyntaxError("no gang partitioning possible in vector-partitioned region")
            elif isinstance(clause,tree.TTAccClauseWorker):
                if resource_filter.worker_partitioned_mode():
                    raise util.eror.SyntaxError("already in worker-partitioned region")
                elif resource_filter.vector_partitioned_mode():
                    raise util.eror.SyntaxError("no worker partitioning possible in vector-partitioned region")
            elif isinstance(clause,tree.TTAccClauseVector):
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

    def _traverse_acc_compute_construct(self,ttaccdir):
        """:note: Syntax checks prevent that num_gangs, num_workers, and
                  vector_length can be specified for TTAccSerial.
        """
        if isinstance(ttaccdir,tree.TTAccSerial):
            self._result.max_num_gangs = "1"    
            self._result.max_num_workers = "1"    
            self._result.max_vector_length = "1"    
        for clause in ttaccdir.walk_clauses_device_type(
            self._result.device_type
          ):
            if isinstance(clause,tree.TTAccClauseNumGangs):
                if clause.arg_specified:
                    self._result.max_num_gangs = clause.arg
            elif isinstance(clause,tree.TTAccClauseNumWorkers):
                if clause.arg_specified:
                    self._result.max_num_workers = clause.arg
            elif isinstance(clause,tree.TTAccClauseVectorLength):
                if clause.arg_specified:
                    self._result.max_vector_length = clause.arg
            elif isinstance(clause,tree.TTAccClausePrivate):
                self._result.private_vars = clause.var_list 
                self._result.generated_code += render.render_private_vars_decl_list(
                  ttvalues,scope
                )
            elif isinstance(clause,tree.TTAccClauseFirstprivate):
                self._result.firstprivate_vars = clause.var_list
        self._result.generated_code += loops.render_hip_kernel_prolog_acc()
        self._compute_construct = ttaccdir 

    def _traverse_values(self,values):
        # overwrite: c_name for derived type members if appropriate
        # TODO outsource this so that we can use same routine in scanner?
        # Or put it into the results 
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
                     if ttfunccall.symbol_info.name == "present":
                         # add optional argument
                         pass
                     elif ttfunccall.is_elemental_call:
                        # todo: Check if we can map the function
                        ttfunccall.overwrite_cstr(
                          "_"+func_name,
                          list(ttfunccall.rank_defining_node.args)
                        )

    def _find_rvalues_in_directive(self,ttnode):
        rvalues = []
        for clause in ttnode.walk_clauses_device_type(
            self._result.device_type
          ):
            if isinstance(clause,(
                tree.TTAccClauseGang,
                tree.TTAccClauseWorker,
                tree.TTAccClauseVector,
                tree.TTAccClauseTile,
              )):
                for child in clause.walk_preorder():
                    if isinstance(child,tree.TTValue):
                        rvalues.append(child)
        self._result.rvalues += rvalues
        self._traverse_values(rvalues)
    
    def _find_lvalues_rvalues_in_arith_expr(self,ttnode):
        """Looks for rvalues and rvalues in an arithmetic expression
        or assignment."""
        lvalues, rvalues = [], []
        for child in ttnode.walk_preorder():
            if isinstance(child,tree.TTRvalue):
                rvalues.append(child)
            elif isinstance(child,tree.TTLvalue):
                lvalues.append(child)
        self._result.lvalues += lvalues
        self._result.rvalues += rvalues
        self._traverse_values(lvalues)
        self._traverse_values(rvalues)
   
    def _traverse_acc_loop_directive(self,ttaccdir):  
        """Create new AccLoopnestManager instance. Append it to the result's list.
        Render it if no collapse or tile clause is specified.
        Search certain clauses for rvalues and lvalues.
        """
        # loop directives might contain rvalues that need to be passed
        self._find_rvalues_in_directive(ttaccdir)
        self._check_loop_parallelism(
          self._resource_filter,
          ttaccdir
        )
        #
        self._loopnest_mgr = loopmgr.AccLoopnestManager()
        for clause in ttaccdir.walk_clauses_device_type(
            self._result.device_type
          ):
            if isinstance(clause,tree.TTAccClauseGang):
                self._loopnest_mgr.gang_specified = True
                self._loopnest_mgr.gang = clause.arg
            elif isinstance(clause,tree.TTAccClauseWorker):
                self._loopnest_mgr.worker_specified = True
                self._loopnest_mgr.worker = clause.arg
            elif isinstance(clause,tree.TTAccClauseVector):
                self._loopnest_mgr.vector_specified = True
                self._loopnest_mgr.vector = clause.arg
            elif isinstance(clause,tree.TTAccClausePrivate):
                self._loopnest_mgr.private = clause.var_list
            elif isinstance(clause,tree.TTAccClauseReduction):
                self._loopnest_mgr.reduction += [
                  (var,clause.op) for var in clause.var_list
                ]
   
    # TODO get rid of info object
    def _traverse_cuf_kernel_do_construct(self,ttnode):
        pass
    #    cuf_construct_info = analysis.cuf.analyze_directive(ttnode)  
    #    if cuf_construct_info.grid.specified:
    #        self._result.grid = cuf_construct_info.grid
    #    if cuf_construct_info.block.specified:
    #        self._result.block = cuf_construct_info.block
    #    if cuf_construct_info.sharedmem.specified:
    #        self._result.sharedmem = cuf_construct_info.sharedmem
    #    if cuf_construct_info.stream.specified:
    #        self._result.stream = cuf_construct_info.stream
    #    if cuf_construct_info.reduction.specified: 
    #        self._result.reductions = cuf_construct_info.reduction
   
    # TODO get rid of info object
    def _traverse_cuf_kernel_do_loop_directive(self,ttdo):
        pass 
    #    """Create new AccLoopnestManager instance. Append it to the result's list.
    #    Render it if no number of loops is specified. 
    #    Search certain clauses for rvalues and lvalues.
    #    """
    #    #todo: split annotation from loop, init AccLoopnestManager solely with acc_loop_info
    #    cuf_loop_info = analysis.cuf.analyze_directive(
    #      ttdo.annotation
    #    ) 
    #    acc_loop_info = analysis.acc.AccLoopInfo(None)
    #    acc_loop_info.gang.specified = True
    #    acc_loop_info.worker.specified = True
    #    acc_loop_info.vector.specified = True
    #    #
    #    if cuf_loop_info.num_loops.value.specified:
    #        acc_loop_info.collapse.value = cuf_loop_info.num_loops.value
    #    self._init_loopnest_mgr(ttdo,acc_loop_info)

    def _traverse_container(self,ttnode):
        """Add container header, increase indent, 
        traverse statements in container body, 
        decrease indent again, add container footer.
        :note: Container statements are never subject to masks."""
        self._find_lvalues_rvalues_in_arith_expr(ttnode)
        self._add_unmasked_code(ttnode.header_cstr())
        previous_indent = self._indent
        self._indent += ttnode.indent
        # descend
        self._traverse_container_body(ttnode)
        # ascend
        self._indent = previous_indent
        self._add_unmasked_code(ttnode.footer_cstr())

    def _unpack_render_result_and_descend(self,ttdo,render_result):
        """Unpack render result, update resource filter, increase indent,
        traverse statements in container body,
        restore previous resource filter, restore previous indent.
        :note: Container statements are never subject to masks."""
        (loopnest_open,
         loopnest_close,
         loopnest_resource_filter,
         loopnest_indent) = render_result
        self._add_unmasked_code(loopnest_open)

        self._resource_filter += loopnest_resource_filter
        previous_indent = self._indent
        self._indent += loopnest_indent
        # descend
        self._traverse_container_body(ttdo,indent="")
        # ascend
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
            if not self._loopnest_mgr.is_complete():
                self._loopnest_mgr.append_do_loop(ttdo)
                if self._loopnest_mgr.is_complete():
                    self._loopnest_mgr.apply_loop_transformations()
                    self._render_loopnest_and_descend(ttdo)
                else:
                    self._traverse_container_body(ttdo,"")
        else:
            render_result = loopmgr.create_simple_loop(
              ttdo).map_to_hip_cpp(self._scope)
            #:todo: fix statement filter
            self._unpack_render_result_and_descend(ttdo,render_result)

    def _create_default_loopnest_mgr_for_array_operation(self,num_collapse):
        """default parallelism for mapping array operations.
           Per default, gang-vector-parallelism is specified.
        """
        if type(self._compute_construct) == tree.TTAccKernels:
            self._loopnest_mgr = loopmgr.AccLoopnestManager(
              gang = None,
              vector = None,
              collapse = num_collapse
            )

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
        elif isinstance(ttnode,tree.TTSubstContainer):
            self._traverse_container_body(ttnode,indent="")
        elif isinstance(ttnode,tree.TTSubstStatement):
            self._traverse(ttnode.subst)
        elif isinstance(ttnode,tree.TTUnrolledArrayAssignment):
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
        self._result.generated_code = self._masked_code_list.render()
        # add the prolog
        return self._result

def map_to_hip_cpp(
    ttroot,
    scope,
    device_type = None
  ):
    """Transform an OpenACC compute construct or routine body to HIP C++.
    :param scope: A scope object, see GPUFORT's indexer component.
    :param str device_type: The device type (`nvidia`, `radeon` or None).
     """
    assert type(ttroot) == tree.TTRoot
    loops.single_level_indent = opts.single_level_indent
    codegen = HIPKernelBodyGenerator()
    codegen.single_level_indent = opts.single_level_indent
    codegen.map_to_flat_arrays = opts.map_to_flat_arrays
    codegen.map_to_flat_scalars = opts.map_to_flat_scalars
    # todo: copy, reduction, hipblas detection must go here?
    assignments.unroll_all_array_assignments(ttroot)
    # insert artificial acc loop node if first compute construct child is
    # a substituted array expression
    for ttstmt in ttroot.body:
        if (
           type(ttstmt) == tree.TTAccKernels
           and type(ttstmt.body[0]) == tree.TTSubstContainer
           and type(ttstmt.body[0].orig) == tree.TTAssignment
          ):
            collapse_expr = tree.TTNumber([
              str(ttstmt.body[0].orig.rank)
            ])
            clauses = [
              tree.TTAccClauseGang([]),
              tree.TTAccClauseVector([]),
              tree.TTAccClauseCollapse([collapse_expr]),
            ]
            ttstmt.body.insert(0,tree.TTAccLoop(clauses))
    # todo: Do arg analyis here
    return codegen.map_to_hip_cpp(
      ttroot,
      scope,
      device_type
    )
