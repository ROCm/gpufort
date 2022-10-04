# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.

class TransformationResult:
    def __init__(self):
        self.grid = None
        self.block = None
        self.stream = None
        self.shared_mem = None
        self.max_num_gangs = None
        self.max_num_workers = None
        self.max_vector_length = None
        self.async_arg = None
        self.loopnest_mgr_list = []
        self.private_variables = []
        self.firstprivate_variables = []
        self.mappings = {}
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
                        default_num_workers, 
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
                        default_num_workers,
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
                        default_num_workers,
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
