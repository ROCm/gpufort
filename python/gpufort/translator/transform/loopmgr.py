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
    
def create_simple_loop(ttdo):
    return loops.Loop(
      index = ttdo.index.cstr(),
      first = ttdo.first.cstr(),
      last = ttdo.last.cstr(),
      step = ttdo.step.cstr() if ttdo.has_step() else None
    )

def create_simple_loop_manager(ttdo):
    loop_mgr = LoopManager()
    loop_mgr.index = ttdo.index
    loop_mgr.first = ttdo.first
    loop_mgr.last = ttdo.last
    if ttdo.has_step():
        loop_mgr.step.value = ttdo.step
    return loop_mgr

class LoopnestManager:
    
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
#        loop = create_simple_loop(ttdo)
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
#        loop_mgr = create_simple_loop_manager(ttdo)
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
    a single acc loop directive (or combined construct) and might only handle
    multiple loops if a tile or collapse clause is specified.
    """
    def __init__(self,**kwargs):
        r"""
        :param \*\*kwargs: Keyword arguments.
        
        **Keyword arguments:**

        * **private_vars (`list`)**: 
            List of variable references tree nodes.
        * **reduction (`list`)**: 
            List of tuples. Each tuple stores a variable reference tree
            node and the associated reduction operator.
        * **collapse**: 
            Arithmetic expression specifying the number of loops to collapse. 
        * **tile (`list`)**: 
            List of arithmetic expressions specifying the tiling to 
            for the next `len(tile)` loops. 
        * **gang (expr)**: 
            Expression for the number of gangs to use, or None (auto).
            Setting this keyword argument to any value implies that there is gang parallelism.
        * **worker (expr)**: 
            Expression for the number of workers to use, or None (auto).
            Setting this keyword argument to any value implies that there is worker parallelism.
        * **vector (expr)**: 
            Expression for the number of vector lanes to use, or None (auto).
            Setting this keyword argument to any value implies that there is vector lane parallelism.
        """
        util.kwargs.set_from_kwargs(self,"private_vars",[],**kwargs)
        util.kwargs.set_from_kwargs(self,"reduction",[],**kwargs)
        (collapse, _) = util.kwargs.get_value("collapse","1",**kwargs)
        self.collapse = self._convert_collapse_expr_to_int(collapse)
        (tile, found) = util.kwargs.get_value("tile",[],**kwargs)
        self.tile = [tree.traversals.make_cstr(e) for e in tile]
        (self.gang, self.gang_specified) = util.kwargs.get_value("gang",None,**kwargs)
        (self.worker, self.worker_specified) = util.kwargs.get_value("worker",None,**kwargs)
        (self.vector, self.vector_specified) = util.kwargs.get_value("vector",None,**kwargs)
        #
        self.loopnest = loops.Loopnest()
        self.loop_mgr_list = [] 

    def is_complete(self):
        """:return: if this loopnest is complete, i.e. can be rendered."""
        assert (self.collapse <= 1 
                or len(self.tile) == 0), "cannot be specified both"
        if len(self.tile):
            assert self.collapse <= 1
            return len(self.tile) == len(self.loopnest)
        else:
            assert not len(self.tile)
            return self.collapse == len(self.loopnest)

    def _create_loop(self,ttdo):
        loop = create_simple_loop(ttdo)
        if self.gang != None:
            loop.num_gangs = tree.traversals.make_cstr(self.gang)
        if self.worker != None:
            loop.num_workers = tree.traversals.make_cstr(self.worker)
        if self.vector != None:
            loop.vector_length = tree.traversals.make_cstr(self.vector)
        loop.gang_partitioned = self.gang_specified
        loop.worker_partitioned = self.worker_specified
        loop.vector_partitioned = self.vector_specified
        return loop         
 
    def _create_loop_manager(self,ttdo):
        loop_mgr = create_simple_loop_manager(ttdo)
        if self.gang_specified:
            loop_mgr.gang.value = self.gang 
        if self.worker_specified:
            loop_mgr.worker.value = self.worker 
        if self.vector_specified:
            loop_mgr.vector.value = self.vector 
        return loop_mgr
    
    def append_do_loop(self,ttdo):
        assert not self.is_complete()
        # use loop information only for first loop
        if not len(self.loop_mgr_list):
            self.loopnest.append(self._create_loop(ttdo))
            self.loop_mgr_list.append(self._create_loop_manager(ttdo))
        else:
            self.loopnest.append(create_simple_loop(ttdo))
            self.loop_mgr_list.append(create_simple_loop_manager(ttdo))

    def _map_loopnest_to_hip_cpp(self,loopnest,
                         num_collapse, # type: int
                         tile_sizes): # type: list[Union[TTNode,str,int]]
        assert self.is_complete()
        if self.collapse > 1:
            return loopnest.collapse().map_to_hip_cpp()
        elif len(self.tile) >= 1:
            return loopnest.tile(self.tile).map_to_hip_cpp()
        else:
            return loopnest.map_to_hip_cpp()

    def map_loopnest_to_hip_cpp(self,scope):
        (loopnest_open,loopnest_close,loopnest_resource_filter,
        loopnest_indent) = self._map_loopnest_to_hip_cpp(
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
            first_loop_mgr = self.loop_mgr_list[0]
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
