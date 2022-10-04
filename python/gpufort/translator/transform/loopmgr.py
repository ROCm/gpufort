# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from .. import tree
from .. import optvals

from . import loops
from . import render

class LoopManager:
    def __init__(self):
        self.index = None
        self.first = None 
        self.end = None 
        self.step = optvals.OptionalSingleValue()
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
        if self.step.specified:
            return "{op}({first},{last},{step})".format(
              operator_loop_len,
              first=converter(self.first),
              last=converter(self.last),
              step=converter(self.step.value)
            )
        else:
            return "{op}({first},{last})".format(
              op=operator_loop_len,
              first=converter(self.first),
              last=converter(self.last)
            )

class LoopnestManager:
    def __init__(self):
        self.loopnest = loops.Loopnest()
        self.loop_mgr_list = [] 
        self.collapse = 0
        self.tile = []
        self.private_vars = []
        self.reductions = {}

    def _create_loop(self,ttdo,acc_loop_info=None):
        if acc_loop_info == None:
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
              num_gangs = acc_loop_info.gang.value,
              num_workers = acc_loop_info.worker.value,
              vector_length = acc_loop_info.vector.value,
              gang_partitioned = acc_loop_info.gang.specified,
              worker_partitioned = acc_loop_info.worker.specified,
              vector_partitioned = acc_loop_info.vector.specified
            )
          
    def _create_loop_manager(self,ttdo,acc_loop_info=None):
        loop_mgr = LoopManager()
        loop_mgr.index = ttdo.index
        loop_mgr.first = ttdo.first
        loop_mgr.last = ttdo.last
        if ttdo.has_step():
            loop_mgr.step.value = ttdo.step
        if acc_loop_info != None:
            loop_mgr.gang.value = acc_loop_info.gang.value 
            loop_mgr.worker.value = acc_loop_info.worker.value 
            loop_mgr.vector.value = acc_loop_info.vector.value 
        return loop_mgr
    
    def append_do_loop(self,ttdo,acc_loop_info=None):
        self.loopnest.append(
          self._create_loop(ttdo,acc_loop_info)
        )
        self.loopnest_mgr_list.append(
          self._create_loop(ttdo,acc_loop_info)
        )

    def _map_loopnest_to_hip_cpp(self,loopnest,
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

    def map_loopnest_to_hip_cpp(self):
        loopnest_open,\
        loopnest_close,\
        loopnest_resource_filter,\
        loopnest_indent =\
          self._map_loopnest_to_hip_cpp(
            self.loopnest,
            self.collapse,
            self.tile
          )
        if len(self.private_vars):
            loopnest_open += textwrap.indent(
              render_private_variables_decl_list(self.private_vars),
              loopnest_indent
            )
        if len(self.reduction):
            first_loop_mgr = self.loopnest.loopnest_mgr_list[0]
            # todo: render reduction variables
            # 1. create unique index var with value = loopnest.index()
            #parallelism = [first_loop_mgr.gang.specified,
            #               first_loop_mgr.worker.specified,
            #               first_loop_mgr.vector.specified]
            #if   parallelism == [False,False,False]:
            #    pass
            #elif parallelism == [False,False,True ]: 
            #    pass
            #elif parallelism == [False,True ,False]: 
            #    pass
            #elif parallelism == [False,True ,True ]:
            #    pass
            #elif parallelism == [True ,False,False]:
            #    pass
            #elif parallelism == [True ,False,True ]: 
            #    pass
            #elif parallelism == [True ,True ,False]: 
            #    pass
            #elif parallelism == [True ,True ,True ]:
            #    pass
            pass
        return (loopnest_open,
                loopnest_close,
                loopnest_resource_filter,
                loopnest_indent)
