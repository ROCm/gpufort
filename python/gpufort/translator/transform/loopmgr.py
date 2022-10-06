# SPDX-License-Identifier: MITc
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import ast
import textwrap

from gpufort import util

from .. import tree
from .. import optvals

from . import loops
from . import render

class LoopManager:
    def __init__(self):
        self.index = None
        self.first = None 
        self.last = None 
        self.step = optvals.OptionalSingleValue()
        self.gang = optvals.OptionalSingleValue()
        self.worker = optvals.OptionalSingleValue()
        self.vector = optvals.OptionalSingleValue()
        self.grid_dim = optvals.OptionalSingleValue()

    def _assert_is_well_defined(self):
        assert self.index != None
        assert self.first != None
        assert self.last != None

    def render_loop_len(self,
                        operator_loop_len,
                        converter):
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
              op=operator_loop_len,
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
    def _create_simple_loop(self,ttdo):
        return loops.Loop(
          index = ttdo.index.cstr(),
          first = ttdo.first.cstr(),
          last = ttdo.last.cstr(),
          step = ttdo.step.cstr() if ttdo.has_step() else None
        )

    def _create_simple_loop_manager(self,ttdo):
        loop_mgr = LoopManager()
        loop_mgr.index = ttdo.index
        loop_mgr.first = ttdo.first
        loop_mgr.last = ttdo.last
        if ttdo.has_step():
            loop_mgr.step.value = ttdo.step
        return loop_mgr
    
    def _convert_collapse_expr_to_int(self,expr):
        try:
            return int(ast.literal_eval(tree.traversals.make_cstr(expr)))
        except:
            raise util.error.LimitationError("expected expression that can be evaluated to int for 'collapse'")

class CufLoopnestManager(LoopnestManager):
    """A loopnest manager is associated with
    a single cuf loop directive and might only contain
    multiple loops if a tile or collapse clause
    is specified.
    """
    def __init__(self,cuf_loop_info=None):
        self.cuf_loop_info = cuf_loop_info
        self.loopnest = loops.Loopnest()
        self.loop_mgr_list = [] 
        self.num_loops

#    def _create_loop(self,ttdo,cuf_loop_info):
#        loop = self._create_simple_loop(ttdo)
#        if cuf_loop_info != None:
#            loop.num_gangs = tree.traversals.make_cstr(cuf_loop_info.gang.value)
#            loop.num_workers = tree.traversals.make_cstr(cuf_loop_info.worker.value)
#            loop.vector_length = tree.traversals.make_cstr(cuf_loop_info.vector.value)
#            loop.gang_partitioned = cuf_loop_info.gang.specified
#            loop.worker_partitioned = cuf_loop_info.worker.specified
#            loop.vector_partitioned = cuf_loop_info.vector.specified
#        return loop         
# 
#    def _create_loop_manager(self,ttdo,cuf_loop_info):
#        loop_mgr = self._create_simple_loop_manager(ttdo)
#        if cuf_loop_info != None:
#            loop_mgr.gang.value = cuf_loop_info.gang.value 
#            loop_mgr.worker.value = cuf_loop_info.worker.value 
#            loop_mgr.vector.value = cuf_loop_info.vector.value 
#        return loop_mgr
#    
#    def append_do_loop(self,ttdo):
#        # use loop information only for first loop
#        if not len(loop_mgr_list):
#            cuf_loop_info = self.cuf_loop_info
#        else:
#            cuf_loop_info = None
#        self.loopnest.append(
#          self._create_loop(ttdo,self.cuf_loop_info)
#        )
#        self.loop_mgr_list.append(
#          self._create_loop_manager(ttdo,self.cuf_loop_info)
#        )
#
#    def _map_loopnest_to_hip_cpp(self,loopnest,
#                         num_collapse, # type: int
#                         tile_sizes): # type: list[Union[TTNode,str,int]]
#        assert num_collapse <= 1 or len(tile_sizes) == 0,\
#         "cannot be specified both"
#        assert num_collapse <= 1 or num_collapse == len(loopnest)
#        assert len(tile_sizes) == 0 or len(tile_sizes) == len(loopnest)
#        if len(loopnest) == num_collapse:
#            return loopnest.collapse().map_to_hip_cpp()
#        elif len(loopnest) == len(tile_sizes):
#            return loopnest.tile(tile_sizes).map_to_hip_cpp()
#        else:
#            return loopnest.map_to_hip_cpp()
#
#    def map_loopnest_to_hip_cpp(self):
#        loopnest_open,\
#        loopnest_close,\
#        loopnest_resource_filter,\
#        loopnest_indent =\
#          self._map_loopnest_to_hip_cpp(
#            self.loopnest,
#            self.collapse,
#            self.tile
#          )
#        if len(self.private_vars):
#            loopnest_open += textwrap.indent(
#              render_private_vars_decl_list(self.private_vars),
#              loopnest_indent
#            )
#        if len(self.reduction):
#            first_loop_mgr = self.loopnest.loop_mgr_list[0]
#            # todo: render reduction variables
#            # 1. create unique index var with value = loopnest.index()
#            #parallelism = [first_loop_mgr.gang.specified,
#            #               first_loop_mgr.worker.specified,
#            #               first_loop_mgr.vector.specified]
#            #if   parallelism == [False,False,False]:
#            #    pass
#            #elif parallelism == [False,False,True ]: 
#            #    pass
#            #elif parallelism == [False,True ,False]: 
#            #    pass
#            #elif parallelism == [False,True ,True ]:
#            #    pass
#            #elif parallelism == [True ,False,False]:
#            #    pass
#            #elif parallelism == [True ,False,True ]: 
#            #    pass
#            #elif parallelism == [True ,True ,False]: 
#            #    pass
#            #elif parallelism == [True ,True ,True ]:
#            #    pass
#            pass
#        return (loopnest_open,
#                loopnest_close,
#                loopnest_resource_filter,
#                loopnest_indent)

class AccLoopnestManager(LoopnestManager):
    """A loopnest manager is associated with
    a single acc loop directive and might only contain
    multiple loops if a tile or collapse clause
    is specified.
    """
    def __init__(self,acc_loop_info=None):
        self.acc_loop_info = acc_loop_info
        self.collapse = 1
        self.tile = []
        self.private_vars = []
        self.reduction = {}
        if acc_loop_info != None:
            if acc_loop_info.private_vars.specified:
                self.private_vars = acc_loop_info.private_vars.value
            if acc_loop_info.reduction.specified:
                pass # todo: loop-wise reductions
            if acc_loop_info.collapse.specified:
                self.collapse = self._convert_collapse_expr_to_int(
                  acc_loop_info.collapse.value
                )
            if acc_loop_info.tile.specified:
                self.tile = [
                  tree.traversals.make_cstr(e) 
                  for e in acc_loop_info.tile.value
                ]
        #
        self.loopnest = loops.Loopnest()
        self.loop_mgr_list = [] 

    def iscomplete(self):
        """:return: if this loopnest is complete, i.e. can be rendered."""
        if len(self.tile):
            assert self.collapse <= 1
            return len(self.tile) == len(self.loopnest)
        else:
            assert not len(self.tile)
            return self.collapse == len(self.loopnest)

    def _create_loop(self,ttdo,acc_loop_info):
        loop = self._create_simple_loop(ttdo)
        if acc_loop_info != None:
            if acc_loop_info.gang.value != None:
                loop.num_gangs = tree.traversals.make_cstr(acc_loop_info.gang.value)
            if acc_loop_info.worker.value != None:
                loop.num_workers = tree.traversals.make_cstr(acc_loop_info.worker.value)
            if acc_loop_info.vector.value != None:
                loop.vector_length = tree.traversals.make_cstr(acc_loop_info.vector.value)
            loop.gang_partitioned = acc_loop_info.gang.specified
            loop.worker_partitioned = acc_loop_info.worker.specified
            loop.vector_partitioned = acc_loop_info.vector.specified
        return loop         
 
    def _create_loop_manager(self,ttdo,acc_loop_info):
        loop_mgr = self._create_simple_loop_manager(ttdo)
        if acc_loop_info != None:
            if acc_loop_info.gang.specified:
                loop_mgr.gang.value = acc_loop_info.gang.value 
            if acc_loop_info.worker.specified:
                loop_mgr.worker.value = acc_loop_info.worker.value 
            if acc_loop_info.vector.specified:
                loop_mgr.vector.value = acc_loop_info.vector.value 
        return loop_mgr
    
    def append_do_loop(self,ttdo):
        assert not self.iscomplete()
        # use loop information only for first loop
        if not len(self.loop_mgr_list):
            acc_loop_info = self.acc_loop_info
        else:
            acc_loop_info = None
        self.loopnest.append(
          self._create_loop(ttdo,self.acc_loop_info)
        )
        self.loop_mgr_list.append(
          self._create_loop_manager(ttdo,self.acc_loop_info)
        )

    def _map_loopnest_to_hip_cpp(self,loopnest,
                         num_collapse, # type: int
                         tile_sizes): # type: list[Union[TTNode,str,int]]
        assert num_collapse <= 1 or len(tile_sizes) == 0,\
         "cannot be specified both"
        assert num_collapse <= 1 or num_collapse == len(loopnest)
        assert len(tile_sizes) == 0 or len(tile_sizes) == len(loopnest)
        if num_collapse > 1 and len(loopnest) == num_collapse:
            return loopnest.collapse().map_to_hip_cpp()
        elif len(tile_sizes) >= 1 and len(loopnest) == len(tile_sizes):
            return loopnest.tile(tile_sizes).map_to_hip_cpp()
        else:
            return loopnest.map_to_hip_cpp()

    def map_loopnest_to_hip_cpp(self,scope):
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
              render.render_private_vars_decl_list(self.private_vars,scope),
              loopnest_indent
            )
        if len(self.reduction):
            first_loop_mgr = self.loopnest.loop_mgr_list[0]
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
