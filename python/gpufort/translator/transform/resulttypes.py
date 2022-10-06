# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.

class TransformationResult:
    def __init__(self,device_type):
        self.device_type = device_type
        self.grid = None
        self.block = None
        self.stream = None
        self.shared_mem = None
        self.max_num_gangs = None
        self.max_num_workers = None
        self.max_vector_length = None
        self.async_arg = None
        self.loopnest_mgr_list = []
        self.private_vars = []
        self.firstprivate_vars = []
        self.mappings = {}
        self.lvalues = []
        self.rvalues = []
        self.generated_code = ""
    @property
    def loop_variables(self):
        result = []
        for loopnest_mgr in self.loopnest_mgr_list:
              for loop_mgr in loopnest_mgr.loop_mgr_list:
                  result.append(loop_mgr.loop_var)
        return result

    def _render_num_gangs_expr(self,
                               loop_len,
                               gang,
                               worker,
                               vector,
                               default_num_workers,
                               default_vector_length,
                               operator_div_round_up, # type: str 
                               converter):
        if gang.value != None:
            return converter(gang.value)
        else:
            if worker.specified and vector.specified:
                return "{op}({loop_len},({num_workers})*({vector_length}))".format(
                  op=operator_div_round_up,
                  loop_len=loop_len,
                  num_workers=converter(
                    worker.value if worker.value != None else default_num_workers
                  ),
                  vector_length=converter(
                    vector.value if vector.value != None else default_vector_length
                  )
                )
            elif worker.specified:
                return "{op}({loop_len},{num_workers})".format(
                  op=operator_div_round_up,
                  loop_len=loop_len,
                  num_workers=converter(
                    worker.value if worker.value != None else default_num_workers
                  )                    )
            elif vector.specified:
                return "{op}({loop_len},({vector_length})".format(
                  op=operator_div_round_up,
                  loop_len=loop_len,
                  vector_length=converter(
                    vector.value if vector.value != None else default_vector_length
                  )
                )
            else:
                return loop_len

    def _render_num_workers_expr(self,
                                 loop_len,
                                 worker,
                                 default_num_workers,
                                 converter):
       if worker.value != None:
           return converter(worker.value)
       else:
           return default_num_workers
    

    def _render_vector_length_expr(self,
                                 loop_len,
                                 vector,
                                 default_vector_length,
                                 converter):
        if vector.value != None:
            return converter(vector.value)
        else:
            return default_vector_length
      
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
        worker_specs = []
        vector_specs = []
        for loopnest_mgr in self.loopnest_mgr_list:
            loop_lens = []
            first_loop_mgr = loopnest_mgr.loop_mgr_list[0]
            for loop_mgr in loopnest_mgr.loop_mgr_list:
                loop_lens.append(loop_mgr.render_loop_len(
                  operator_loop_len,
                  converter
                ))
            if len(loop_lens) == 1:
                loop_len = loop_lens[0]
            else:
                loop_len = "*".join(
                  ["({})".format(l) for l in loop_lens]
                )
            if first_loop_mgr.gang.specified:
                new = self._render_num_gangs_expr(
                  loop_len,
                  first_loop_mgr.gang,
                  first_loop_mgr.worker,
                  first_loop_mgr.vector,
                  default_num_workers,
                  default_vector_length,
                  operator_div_round_up,
                  converter
                )
                if new not in grid_specs:
                    grid_specs.append(new)
            if first_loop_mgr.worker.specified:
                new = self._render_num_workers_expr(
                  loop_len,
                  first_loop_mgr.worker,
                  default_num_workers,
                  converter
                )
                if new not in worker_specs:
                    worker_specs.append(new)
                    
            if first_loop_mgr.vector.specified:
                new = self._render_vector_length_expr(
                  loop_len,
                  first_loop_mgr.vector,
                  default_vector_length,
                  converter
                )
                if new not in vector_specs:
                    vector_specs.append(new)
        if self.grid != None:
            grid = converter(self.grid)
        elif self.max_num_gangs != None:
            grid = converter(self.max_num_gangs)
        elif len(grid_specs) > 1:
            grid = "{op}({args})".format(
              op=operator_max, 
              args=",".join(grid_specs)
            )
        elif len(grid_specs) == 1:
            grid = grid_specs[0]
        else:
            grid = "1"
        if self.block != None:
            block = converter(self.block)
        elif ( self.max_num_workers != None
             and self.max_vector_length != None ):
            block = [
              converter(self.max_vector_length),
              converter(self.max_num_workers)
            ]
        else:
            block = []
            if len(vector_specs) > 1:
                block.append(
                  "{op}({args})".format(
                      op=operator_max, 
                      args=",".join(vector_specs)
                  )
                ) 
            elif len(vector_specs) == 1:
                block.append(
                  vector_specs[0] 
                ) 
            else:
                block.append(
                  converter(default_vector_length)
                ) 
            if len(worker_specs) > 1:
                block.append(
                  "{op}({args})".format(
                      op=operator_max, 
                      args=",".join(worker_specs)
                  )
                ) 
            elif len(worker_specs) == 1:
                block.append(
                  worker_specs[0] 
                ) 
            else:
                block.append(
                  converter(default_num_workers)
                ) 
        return (grid, block)  
