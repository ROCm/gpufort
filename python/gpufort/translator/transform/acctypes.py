# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from .. import tree
from .. import optvals

class LoopInfo(self):
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

class LoopnestInfo:
    def __init__(self):
        self.loopnest = None
        self.loop_info_list = [] 
        self.num_collapse = 0
        self.tile_sizes = []
        self.loop_vars = []
        self.private_vars = []
        self.reductions = []
        self.problem_sizes = []
    def reset(self):
        self.__init__()

class TransformationResult:
    def __init__(self):
        self.max_num_gangs = None
        self.max_num_workers = None
        self.max_vector_length = None
        self.loopnest_info_list = []
        self.private_variables = []
        self.firstprivate_variables = []
        self.mappings = []
        self.lvalues = []
        self.rvalues = []
        self.generated_code = ""
    #self.all_num_gangs = []
    #self.all_num_workers = []
    #self.all_vector_length = []
    #self.all_gang_problem_sizes = []
    #self.all_worker_problem_sizes = []
    #self.all_vector_problem_sizes = []
    #self.all_num_workers = []
    #self.all_vector_length = []
    #self.loop_variables = []
    @property
    def loop_variables(self):
        result = []
        for nest_info in self.loopnest_info_list:
              for loop_info in nest_info.loop_info_list:
                  result.append(loop_info.loop_var)
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
        for nest_info in self.loopnest_info_list:
            for loop_info in nest_info.loop_info_list:
                loop_len = loop_info.render_loop_len(
                  operator_loop_len,
                  converter
                )
                if ( loop_info.gang.specified
                     and not loop_info.worker.specified
                     and not loop_info.vector.specified ):
                    # gang
                    grid_specs.append(
                      _render_grid_size_expr(
                        loop_len,loop_info.gang.value,
                        None,default_num_workers,converter   
                      ))
                elif ( loop_info.gang.specified
                       and     loop_info.worker.specified
                       and not loop_info.vector.specified ):
                    # gang worker
                    grid_specs.append(
                      _render_grid_size_expr(
                        loop_len,
                        loop_info.gang.value,
                        loop_info.worker.value,
                        default_num_workers,
                        converter   
                      ))
                    block_specs.append(
                      _render_block_size_expr(
                        loop_info.worker.value,
                        None,
                        default_num_workers
                        default_vector_length,
                        converter   
                      ))
                elif ( loop_info.gang.specified
                       and     loop_info.vector.specified ):
                    # gang vector and gang worker vector
                    grid_specs.append(
                      _render_grid_size_expr(
                        loop_len,
                        loop_info.gang.value,
                        loop_info.worker.value,
                        default_num_workers,
                        converter   
                      ))
                    block_specs.append(
                      _render_block_size_expr(
                        loop_info.worker.value,
                        loop_info.vector.value,
                        default_num_workers
                        default_vector_length,
                        converter   
                      ))
                elif ( loop_info.worker.specified 
                       or loop_info.vector.specified ):
                    # worker, vector, and worker vector
                    block_specs.append(
                      _render_block_size_expr(
                        loop_info.worker.value,
                        loop_info.vector.value,
                        default_num_workers
                        default_vector_length,
                        converter   
                      ))
        if self.max_num_gangs != None:
            grid = converter(self.max_num_gangs)
        else:
            grid = "{op}({args})".format(
              op=operator_max, 
              args=",".join(grid_specs)
            ) 
        if ( self.max_num_workers != None and
             and self.max_vector_length != None ):
            block = self._render_block_size_expr(
              loop_len,
              num_workers,
              vector_length,
              default_num_workers,
              default_vector_length,
              converter
            )
        return (grid, block)  
