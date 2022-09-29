# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from .. import opts
from .. import analysis
from .. import tree
from .. import optvals

from . import loops
from . import loopmgr

loops.single_level_indent = opts.single_level_indent

class Transformer:
    def __init__(self):
        # traversal state: 
        self._indent
        self._resource_filter
        self._result
        self._loopnest_mgr

    def _check_loop_parallelism(resource_filter,acc_loop_info)
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
            
    def _create_acc_resource_filter(
        self,
        initially_gang_partitioned,
        initially_worker_partitioned,
        initially_vector_partitioned
    ):
        """Initialize gang, worker, vector resources with
        None for all initially partitioned resource loopmgr.
        This indicates that all available resources can be used
        for that resource type. In particular, this is
        used when translating OpenACC routines.
        """
        return loops.AccResourceFilter(
          num_gangs = [None] if initially_gang_partitioned else [],
          num_workers = [None] if initially_worker_partitioned else [],
          vector_length = [None] if initially_vector_partitioned else []
        )
    
    def _traverse_container_body(self,ttcontainer):
        """Traverses a container's children and 
        applies the resource filter's statement
        selection filter to contiguous groups
        of non-container statements.
        """
        previous_self._indent = self._indent
        self._indent += ttcontainer.self._indent
        previous_self._indent2 = self._indent

        statement_selector_is_open = False
        num_children = len(ttcontainer)
        for i,child in enumerate(ttcontainer):
            if isinstance(child,tree.TTContainer):
                if statement_selector_is_open:
                    self._indent = previous_self._indent2
                    self._result.generated_code += self._indent+"}\n"
                    traverse_(child)
            else:
                if ( not statement_selector_is_open
                     and not self._resource_filter.worker_and_vector_partitioned_mode()
                     and not i == num_children-1 ):
                    self._result.generated_code += "{}if ({}) {{".format(
                      self._indent,
                      self._resource_filter.statement_selection_condition()
                    )
                    statement_selector_is_open = True
                    self._indent += opts.single_level_self._indent
                 traverse_(child)
        if statement_selector_is_open:
            self._result.generated_code += self._indent+"}\n"
        self._indent = previous_self._indent

    def _traverse_acc_compute_construct(self,ttnode):
        acc_construct_info = analysis.acc.analyze_directive(ttnode)  
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
        if acc_construct_info.private_variables.specified: 
            self._result.private_variables = acc_construct_info.private_variables
            self._result.generated_code += render.render_private_variables_decl_list(
              ttvalues,scope
            )
        if acc_construct_info.firstprivate_variables.specified: 
            self._result.firstprivate_variables = acc_construct_info.firstprivate_variables
    
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
        if cuf_construct_info.reductions.specified: 
            self._result.reductions = cuf_construct_info.reductions

    def _render_loopnest_mgr_and_descend(self,ttnode):
        loopnest_render_result = self._loopnest_mgr.render_loopnest()
        self._result.append(self._loopnest_mgr)
        self._loopnest_mgr = None
        # 
        loopnest_open,loopnest_close,\
        loopnest_resource_filter,loopnest_indent =\
          loopnest_render_result
        self._result.generated_code += textwrap.indent(
          ttnode.header_cstr(),
          self._indent
        )
        self._resource_filter += loopnest_resource_filter
        previous_indent = self._indent
        self._indent += loopnest_indent
        #
        self._traverse_container_body(ttnode)
        #
        self._resource_filter -= loopnest_resource_filter
        self._indent = previous_indent
  
    def _traverse_acc_loop(self,ttdo):  
        """
        Create new LoopnestManager instance. Append it to the result's list.
        Render it if no collapse or tile clause is specified.
        Search certain clauses for rvalues and lvalues.
        """
        #todo: split annotation from loop, init LoopnestManager with acc_loop_info
        _check_loop_parallelism(
          self._resource_filter,
          acc_loop_info
        )
        acc_loop_info = analysis.analyze_directive(
          ttdo.annotation,
          device_type
        ) 
        if acc_loop_info.private_vars.specified:
            self._loopnest_mgr.private_vars = acc_loop_info.private_vars.value
        if acc_loop_info.reductions.specified:
            pass # todo: loop-wise reductions
        self._loopnest_mgr = loopmgr.LoopnestManager()
        self._loopnest_mgr.append_do_loop(ttdo,acc_loop_info)
        if ( num_collapse == 0
             and not len(tile_sizes) ):
            loopnest_render_result = self._loopnest_mgr.render_loopnest()
            self._result.append(self._loopnest_mgr)
            self._loopnest_mgr = None
        else:
            loopnest_render_result = None
        # loop directives might contain lvalues, rvalues that need to be passed
        analysis.acc.find_lvalues_and_rvalues_in_directive(
          ttdo.annotation,
          self._result.lvalues,
          self._result.rvalues
        )
        return loopnest_render_result
 
    def _traverse_do_loop(self,ttnode):
        # todo: when collapsing, check if we can move statement into inner loop, should be possible if 
        # a loop statement is not relying on the lvalue or inout arg from a previous statement (difficult to analyze)
        # alternative interpretation of collapse user information -> we can reorder statements without error
        render_result = None 
        if ( self._loopnest_mgr != None 
             and self._loopnest_mgr.num_collapse > 1 ):
            self._loopnest_mgr.append_do_loop(ttnode)
            if len(loopnest) == num_collapse:
                render_result = self._loopnest_mgr.render_loopnest()
                self._result.append(self._loopnest_mgr)
                self._loopnest_mgr = None
            return
        elif ( self._loopnest_mgr != None 
               and len(tile_sizes) > 0 ):
            self._loopnest_mgr.append_do_loop(ttnode)
            if len(loopnest) == len(tile_sizes):
                render_result = self._loopnest_mgr.render_loopnest()
                self._result.append(self._loopnest_mgr)
                self._loopnest_mgr = None
            _traverse_container_body(ttnode)
        elif isinstance(ttnode.annotation,tree.TTAccLoop):
            render_result = self._traverse_acc_loop(ttnode)
        else:
            self._loopnest_mgr = loopmgr.LoopnestManager()
            self._loopnest_mgr.append_do_loop(ttnode)
            render_result = self._loopnest_mgr.render_loopnest()
        loopnest_open,loopnest_close,\
        loopnest_resource_filter,loopnest_indent =\
          render_result
        self._result.generated_code += textwrap.indent(
          ttnode.header_cstr(),
          self._indent
        )
        self._resource_filter += loopnest_resource_filter
        previous_indent = self._indent
        self._indent += loopnest_indent
        #
        self._traverse_container_body(ttnode)
        #
        self._resource_filter -= loopnest_resource_filter
        self._indent = previous_indent
    
    def _traverse(self,ttnode):
        lvalues, rvalues = [], []
        if isinstance(ttnode,(
            tree.TTAccParallel,tree.TTAccKernels,
            tree.TTAccParallelLoop,tree.TTAccKernelsLoop
        )): #todo: detach loop annotation from do loop
            self._traverse_acc_compute_construct(ttnode)
        elif isinstance(ttnode,TTDo):
            self._traverse_do_loop(ttnode)
        elif isinstance(ttnode,TTContainer):
            analysis.fortran.find_lvalues_and_rvalues(
              ttnode,
              lvalues,
              rvalues
            )
            #todo: modify expressions
            self._result.generated_code += textwrap.indent(
              ttnode.header_cstr(),
              self._indent
            )
            self._indent += ttnode.self._indent
            #
            self._traverse_container_body(ttnode)
            #
            self._indent = previous_indent
            self._result.generated_code += textwrap.indent(
              ttnode.footer_cstr(),
              self._indent
            )
        else: # other statements
            analysis.fortran.find_lvalues_and_rvalues(
              ttnode,
              lvalues,
              rvalues
            )
            #todo: expand array assignment expressions
            #todo: modify expressions
            self._result.generated_code += ttnode.cstr().rstrip("\n")+"\n"

    def transform_compute_construct(
        self
        ttcomputeconstruct,
        scope,
        device_type = None, 
        initially_gang_partitioned = False,
        initially_worker_partitioned = False,
        initially_vector_partitioned = False
      ):
        """Transform an OpenACC compute construct or routine body to HIP C++.
        :param scope: A scope object, see GPUFORT's indexer component.
        :param str device_type: The device type (`acc_device_nvidia`, `acc_device_radeon` or None).
        :param bool initially_gang_partitioned: Start in gang-partitioned mode.
        :param bool initially_worker_partitioned: Start in worker-partitioned mode.
        :param bool initially_vector_partitioned: Start in vector-partitioned mode.
        """
        loops.reset() # reset variable counters of loops package
        self._result = TransformationResult()
        self._resource_filter = _create_acc_resource_filter(
          initially_gang_partitioned,
          initially_worker_partitioned,
          initially_vector_partitioned)
        self._indent = ""
        self._loopnest_mgr = None
        self._traverse(ttcomputeconstruct)
        return self._result
