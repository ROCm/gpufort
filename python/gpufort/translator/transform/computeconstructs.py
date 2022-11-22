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

class __HIPKernelBodyGenerator:

    def __init__(self):
        self.single_level_indent = "" 
        # traversal state: 
        self._indent = ""
        self._scope = None
        self._resource_filter = None
        self._result = None
        self._loopnest_mgr = None

    def _check_loop_parallelism(self,
          resource_filter,
          acc_loop_info):
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
            self._indent += ""
        else:
            self._indent += ttcontainer.indent
        previous_indent2 = self._indent

        statement_selector_is_open = False
    
        def do_not_mask_(ttnode):
            return isinstance(ttnode,
                    (tree.FlowStatementMarker,
                    tree.TTBlank,
                    tree.TTAccDirectiveBase,
                    tree.TTCufKernelDo,
                    tree.TTCommentedOut))
    
        def close_statement_selector_():
            nonlocal statement_selector_is_open
            nonlocal previous_indent2
            self._indent = previous_indent2
            self._result.generated_code += self._indent+"}\n"
            statement_selector_is_open = False

        num_children = len(ttcontainer)
        for i,child in enumerate(ttcontainer):
            if isinstance(child,tree.TTContainer):
                if statement_selector_is_open:
                    close_statement_selector_()
                self._traverse(child)
            else:
                if ( statement_selector_is_open
                     and do_not_mask_(child) 
                   ):
                      close_statement_selector_()
                elif ( 
                  not statement_selector_is_open
                  and not self._resource_filter.worker_and_vector_partitioned_mode()
                  and not do_not_mask_(child) 
                ):
                    self._result.generated_code += textwrap.indent(
                      "if ( {} ) {{\n".format(
                        self._resource_filter.statement_selection_condition()
                      ),
                      self._indent
                    )
                    statement_selector_is_open = True
                    self._indent += opts.single_level_indent
                self._traverse(child)
        if statement_selector_is_open:
            close_statement_selector_()
        self._indent = previous_indent

    def _visit_acc_compute_construct(self,ttnode):
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
   
    def _visit_acc_loop_directive(self,ttnode):  
        """Create new AccLoopnestManager instance. Append it to the result's list.
        Render it if no collapse or tile clause is specified.
        Search certain clauses for rvalues and lvalues.
        """
        #
        acc_loop_info = analysis.acc.analyze_directive(
          ttnode,
          self._result.device_type
        ) 
        # loop directives might contain lvalues, rvalues that need to be passed
        analysis.acc.find_rvalues_in_directive(
          ttnode,
          self._result.rvalues
        )
        #todo: split annotation from loop, init AccLoopnestManager solely with acc_loop_info
        self._check_loop_parallelism(
          self._resource_filter,
          acc_loop_info
        )
        #
        self._loopnest_mgr = loopmgr.AccLoopnestManager(acc_loop_info)
    
    def _visit_cuf_kernel_do_construct(self,ttnode):
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
    
    def _visit_cuf_kernel_do_loop_directive(self,ttdo):  
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
        analysis.fortran.find_lvalues_and_rvalues(
          ttnode,
          self._result.lvalues,
          self._result.rvalues
        )
        #todo: modify expressions
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
        self._result.generated_code += textwrap.indent(
          loopnest_open,
          self._indent
        )
        self._resource_filter += loopnest_resource_filter
        previous_indent = self._indent
        self._indent += loopnest_indent
        #
        self._traverse_container_body(ttdo,indent="")
        # 
        self._resource_filter -= loopnest_resource_filter
        self._indent = previous_indent
        self._result.generated_code += textwrap.indent(
          loopnest_close,
          self._indent
        )
 
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
            # TODO fix statement filter
            self._unpack_render_result_and_descend(ttdo,render_result)


    def _traverse_statement(self,ttnode):
        analysis.fortran.find_lvalues_and_rvalues(
          ttnode,
          self._result.lvalues,
          self._result.rvalues
        )
        #todo: expand array assignment expressions
        #if isinstance(ttnode,tree.TTAssignment):
        #    if arrayexpr.is_array_assignment(ttnode._lhs,self._scope):
        #        pass
        # todo: modify expressions
        self._result.generated_code += textwrap.indent(
          ttnode.cstr().rstrip("\n")+"\n",
          self._indent
        )
    
    def _traverse(self,ttnode):
        #todo: detach loop annotation from do loop
        if isinstance(ttnode,(tree.TTAccParallelLoop,tree.TTAccKernelsLoop)):
            self._visit_acc_compute_construct(ttnode)
            self._visit_acc_loop_directive(ttnode)
        elif isinstance(ttnode,(tree.TTAccParallel,tree.TTAccKernels)):
            self._visit_acc_compute_construct(ttnode)
        elif isinstance(ttnode,tree.TTCufKernelDo):
            self._visit_cuf_kernel_do_construct(ttnode)
        elif isinstance(ttnode,tree.TTAccLoop):
            if self._loopnest_mgr == None:
                self._visit_acc_loop_directive(ttnode)
        elif isinstance(ttnode,tree.TTDo):
            self._traverse_do_loop(ttnode)
        elif isinstance(ttnode,tree.TTContainer):
            self._traverse_container(ttnode)
        else: # other statements
            self._traverse_statement(ttnode)

    def map_to_hip_cpp(
          self,
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
    return __instance.map_to_hip_cpp(
      ttcomputeconstruct,
      scope,
      device_type
    )
