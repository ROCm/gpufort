# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from .. import tree
from .. import optvals

from . import loops
from . import render

class LoopManager(self):
    def __init__(self):
        self.index = None
        self.first = None 
        self.end = None 
        self.step = None 
        self.gang = optvals.OptionalSingleValue()
        self.worker = optvals.OptionalSingleValue()
        self.vector = optvals.OptionalSingleValue()

    def _assert_is_well_defined(self):
        assert self.index != None
        assert self.first != None
        assert self.end != None

    def render_loop_len(self,
                        converter,
                        operator_loop_len):
        """:return: Expression to compute the length of the loop.
        :param str operator_loop_len: The function to use
                                     for calculating the loop length.
        :param converter: One of tree.traversals.make_fstr or ~.make_cstr
        :note: As the step size can be a runtime value,
               one requires a Fortran/C++ function to compute it.
        """
        self._assert_is_well_defined()
        if self.step == None:
            return "{op}({first},{last})".format(
              op=operator_loop_len,
              first=converter(self.first),
              last=converter(self.last),
              step=converter(self.step)
            )
        else:
            return "{op}({first},{last},{step})".format(
              operator_loop_len,
              first=converter(self.first),
              last=converter(self.last),
              step=converter(self.step)
            )

class LoopnestManager:
    def __init__(self):
        self.loopnest = loops.Loopnest()
        self.loop_mgr_list = [] 
        self.num_collapse = 0
        self.tile_sizes = []
        self.loop_vars = []
        self.private_vars = []
        self.reductions = []
        self.problem_sizes = []
    def _create_loop(self,ttdo,acc_loop_mgr=None):
        if acc_loop_mgr == None:
            return loops.Loop(
              index = ttdo.index.cstr(),
              first = ttdo.first.cstr(),
              last = ttdo.last.cstr(),
              step = ttdo.step.cstr() if ttdo.has_step() else None,
            )
        else:
            return loops.Loop(
              index = ttdo.index.cstr(),
              first = ttdo.first.cstr(),
              last = ttdo.last.cstr(),
              step = ttdo.step.cstr() if ttdo.has_step() else None,
              num_gangs = acc_loop_mgr.gang.value,
              num_workers = acc_loop_mgr.worker.value,
              vector_length = acc_loop_mgr.vector.value,
              gang_partitioned = acc_loop_mgr.gang.specified,
              worker_partitioned = acc_loop_mgr.worker.specified,
              vector_partitioned = acc_loop_mgr.vector.specified
            )
          
    def _create_loop_manager(self,ttdo,acc_loop_mgr=None):
        loop_mgr = LoopManager()
        loop_mgr.index = ttdo.index
        loop_mgr.first = ttdo.first
        loop_mgr.last = ttdo.last
        loop_mgr.step = ttdo.step if ttdo.has_step() else None
        if acc_loop_mgr != None:
            loop_mgr.gang.value = acc_loop_mgr.gang.value 
            loop_mgr.worker.value = acc_loop_mgr.worker.value 
            loop_mgr.vector.value = acc_loop_mgr.vector.value 
        return loop_mgr
    
    def append_do_loop(self,ttdo,acc_loop_mgr=None):
        self.loopnest.append(
          self._create_loop(ttdo,acc_loop_mgr)
        )
        self.loopnest_mgr_list.append(
          self._create_loop(ttdo,acc_loop_mgr)
        )

    def _render_loopnest(self,loopnest,
                         num_collapse, # type: int
                         tile_sizes): # type: list[Union[TTNode,str,int]]
        assert num_collapse <= 1 or len(tile_sizes) == 0,\
         "cannot be specified both"
        assert num_collapse <= 1 or num_collapse == len(loopnest)
        assert len(tile_sizes) == 0 or len(tile_sizes) == len(loopnest)
        if len(loopnest) == num_collapse:
            return loopnest.collapse().map_to_hip_cpp()
        elif len(loopnest) == len(tile_sizes):
            return loopnest.tile(tile_sizes).map_to_hip_cpp()
        else:
            return loopnest.map_to_hip_cpp()

    def render_loopnest(self):
        loopnest_open,\
        loopnest_close,\
        loopnest_resource_filter,\
        loopnest_indent =\
          render.render_loopnest(
            self.loopnest,
            self.num_collapse,
            self.tile_sizes
          )
        if len(private_vars):
            loopnest_open += textwrap.indent(
              render_private_variables_decl_list(private_vars,
              loopnest_indent
            )
            # TODO render reduction variables
            # 1. create unique index var with value = loopnest.index()
            # 2. 
        return (loopnest_open,
                loopnest_close,
                loopnest_resource_filter,
                loopnest_indent)

class TransformationResult:
    def __init__(self):
        self.grid = None
        self.block = None
        self.stream = None
        self.shared_mem = None
        self.max_num_gangs = None
        self.max_num_workers = None
        self.max_vector_length = None
        self.async = None
        self.loopnest_mgr_list = []
        self.private_variables = []
        self.firstprivate_variables = []
        self.mappings = []
        self.lvalues = []
        self.rvalues = []
        self.generated_code = ""
    @property
    def loop_variables(self):
        result = []
        for nest_info in self.loopnest_mgr_list:
              for loop_mgr in nest_info.loop_mgr_list:
                  result.append(loop_mgr.loop_var)
        return result

    def _render_grid_size_expr(self,
                               loop_len,
                               num_gangs,
                               num_workers,
                               default_num_workers,
                               converter):
        if num_gangs != None:
            return converter(num_gangs)
        else:
            "{loop_len}/({num_workers})".format(
              loop_len=loop_len,
              num_workers=converter(
                num_workers if num_workers != None else default_num_workers
              ),
            )
    
    def _render_block_size_expr(self,
                                num_workers,
                                vector_length,
                                default_num_workers,
                                default_vector_length,
                                converter):
        return  "({num_workers})*({vector_length})".format(
          num_workers=converter(
            num_workers if num_workers != None else default_num_workers
          ),
          vector_length=converter(
            vector_length if vector_length != None else default_vector_length
          ),
        )

    def hip_grid_and_block_as_str(
        self,
        default_num_workers,
        default_vector_length,
        operator_max, # type: str
        operator_loop_len, # type: str
        operator_div_round_up, # type: str 
        converter
    ):
        """
        :return: An expression for the 1-dimensional OpenACC compute grid
                 and block so that it can be mapped to CUDA/HIP.
        :default_num_workers:
        :note: Due to the usage of AccResourceFilter in the transformation routine, we
               prevent that parallelism levels
               cannot be prescribed twice in a loopnest and
               a nest of loopnests. Therefore, the transformation
               result does not need to implement such checks.
        """
        grid = None
        block = None
        workers = None
        #
        grid_specs = []
        block_specs = []
        for nest_info in self.loopnest_mgr_list:
            for loop_mgr in nest_info.loop_mgr_list:
                loop_len = loop_mgr.render_loop_len(
                  operator_loop_len,
                  converter
                )
                if ( loop_mgr.gang.specified
                     and not loop_mgr.worker.specified
                     and not loop_mgr.vector.specified ):
                    # gang
                    grid_specs.append(
                      _render_grid_size_expr(
                        loop_len,loop_mgr.gang.value,
                        None,default_num_workers,converter   
                      ))
                elif ( loop_mgr.gang.specified
                       and     loop_mgr.worker.specified
                       and not loop_mgr.vector.specified ):
                    # gang worker
                    grid_specs.append(
                      _render_grid_size_expr(
                        loop_len,
                        loop_mgr.gang.value,
                        loop_mgr.worker.value,
                        default_num_workers,
                        converter   
                      ))
                    block_specs.append(
                      _render_block_size_expr(
                        loop_mgr.worker.value,
                        None,
                        default_num_workers
                        default_vector_length,
                        converter   
                      ))
                elif ( loop_mgr.gang.specified
                       and     loop_mgr.vector.specified ):
                    # gang vector and gang worker vector
                    grid_specs.append(
                      _render_grid_size_expr(
                        loop_len,
                        loop_mgr.gang.value,
                        loop_mgr.worker.value,
                        default_num_workers,
                        converter   
                      ))
                    block_specs.append(
                      _render_block_size_expr(
                        loop_mgr.worker.value,
                        loop_mgr.vector.value,
                        default_num_workers
                        default_vector_length,
                        converter   
                      ))
                elif ( loop_mgr.worker.specified 
                       or loop_mgr.vector.specified ):
                    # worker, vector, and worker vector
                    block_specs.append(
                      _render_block_size_expr(
                        loop_mgr.worker.value,
                        loop_mgr.vector.value,
                        default_num_workers
                        default_vector_length,
                        converter   
                      ))
        if self.grid != None:
            grid = converter(self.grid)
        elif self.max_num_gangs != None:
            grid = converter(self.max_num_gangs)
        else:
            grid = "{op}({args})".format(
              op=operator_max, 
              args=",".join(grid_specs)
            ) 
        if self.block != None:
            block = converter(self.block)
        elif ( self.max_num_workers != None
             and self.max_vector_length != None ):
            block = self._render_block_size_expr(
              loop_len,
              self.max_num_workers,
              self.max_vector_length,
              None,
              None,
              converter
            )
        else:
            block = "{op}({args})".format(
              op=operator_max, 
              args=",".join(block_specs)
            ) 
        return (grid, block)  
