# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from .. import opts
from .. import analysis
from .. import tree
from .. import optvals

from . import loops

loops.single_level_indent = opts.single_level_indent

class LoopInfo(self):
    def __init__(self):
        self.index = None
        self.first = None 
        self.end = None 
        self.step = None 
        self.gang = optvals.OptionalSingleValue()
        self.worker = optvals.OptionalSingleValue()
        self.vector_length = optvals.OptionalSingleValue()

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

class TransformationInfo:
    def __init__(self):
        self.max_num_gangs = None
        self.max_num_workers = None
        self.max_vector_length = None
        self.loopnest_info_list = []
        #self.all_num_gangs = []
        #self.all_num_workers = []
        #self.all_vector_length = []
        #self.all_gang_problem_sizes = []
        #self.all_worker_problem_sizes = []
        #self.all_vector_problem_sizes = []
        #self.all_num_workers = []
        #self.all_vector_length = []
        #self.loop_variables = []
        self.private_variables = []
        self.firstprivate_variables = []
        self.mappings = []
        self.lvalues = []
        self.rvalues = []
        self.generated_code = ""
    @property
    def loop_variables(self):
        result = []
        for ln_info in self.loopnest_info_list:
              for l_info in ln_info.loop_info_list:
                  result.append(l_info.loop_var)
        return result

    def _generator_loops(self):
        for ln_info in self.loopnest_info_list:
              for l_info in ln_info.loop_info_list:
                    yield l_info

    def grid_and_block_as_str(
        self,
        default_num_workers,
        default_vector_length,
        operator_max,
        operator_looplen,
        operator_div_round_up,
        converter=tree.traversals.make_fstr
    ):
        """
        """
        grid = None
        block = None
        workers = None
        if self.max_num_gangs != None:
            grid = converter(self.max_num_gangs)
        else:
            grid_specs = []
            for l_info in self._generator_loops():
                if l_info.gang.specified:
                    if l_info.gang.value != None:
                        grid_specs.append(
                          converter(l_info.gang.value)
                        )
                    else l_info.step != None:
                        grid_specs.append(
                          "{}({},{})",
                          operator_looplen,
                          converter(l_info..value)
                        )
                    else l_info.step != None:
                       
        if (self.max_num_workers != None
            and self.max_vector_length != None):
            block = 
            if grid != None:
                return (grid, block)  

 
# determine problem size and finest granularity of parallelism
# in case of worker/gang_worker parallelism, we need to launch a full warp
# of threads and mask out all threads except warp thread 0
# in case of gang parallelism, we need to launch a full thread block
# of threads and mask out all threads except thread 0
# Problems:
# - size of the outer loop thread groups depends on the size of the inner loops
#   - only have this information on the way up
# - inner loop might be greater than outer loop, so need to be blocked 
# - need to know default vector length, make available
# Considerations:
#   No warp/worker synchronization needed as vector lanes in warp run in lockstep
#   No intergang synchronization needed as gangs run independently from each other
#   Need to distinguish between loops and other statements to put if ( thread id == 0 ) regions
# Identify workshare such as min,max,etc.

def _check_partitioning_mode(resource_filter,loop_parallelism)
    if loop_parallelism.gang_partitioned_mode():
        if resource_filter.gang_partitioned_mode():
            raise util.eror.SyntaxError("already in gang-partitioned region")
        elif resource_filter.worker_partitioned_mode():
            raise util.eror.SyntaxError("no gang partitioning possible in worker-partitioned region")
        elif resource_filter.vector_partitioned_mode():
            raise util.eror.SyntaxError("no gang partitioning possible in vector-partitioned region")
        # TODO get num gangs
    if loop_parallelism.worker_partitioned_mode():
        if resource_filter.worker_partitioned_mode():
            raise util.eror.SyntaxError("already in worker-partitioned region")
        elif resource_filter.vector_partitioned_mode():
            raise util.eror.SyntaxError("no worker partitioning possible in vector-partitioned region")
    if loop_parallelism.vector_partitioned_mode():
        if resource_filter.vector_partitioned_mode():
            raise util.eror.SyntaxError("already in vector-partitioned region")
        
def _init_acc_resource_filter(
    initially_gang_partitioned = False,
    initially_worker_partitioned = False,
    initially_vector_partitioned = False):
  resource_filter = loops.AccResourceFilter(
    num_gangs = [None] if initially_gang_partitioned else [],
    num_workers = [None] if initially_worker_partitioned else [],
    vector_length = [None] if initially_vector_partitioned else []
  )

def _create_unpartitioned_loop(ttdo):
    return loops.Loop(
      index = ttnode.loop_var(),
      first = ttnode.begin_cstr(),
      last = ttnode.end_cstr(),
      step = ttnode.step_cstr() if ttnode.has_step() else None
    )

def transform_acc_construct(ttaccconstruct,
                            scope,
                            device_types, 
                            initially_gang_partitioned = False,
                            initially_worker_partitioned = False,
                            initially_vector_partitioned = False):
    """
    """
    results = [] # one per device spec
    #device_types = identify_device_types(ttaccconstruct)
    
    for device_type in device_types:
        # for the whole compute construct
        trafo_info = TransformationInfo()
        trafo_info.generated_code = ""
        resource_filter = _init_acc_resource_filter(
          initially_gang_partitioned,
          initially_worker_partitioned,
          initially_vector_partitioned)
        indent = ""
        #
        # per loopnest in the compute construct
        ln_info = LoopnestInfo()
        def traverse_container_body_(ttcontainer):
            nonlocal trafo_info
            nonlocal indent
            nonlocal resource_filter
            previous_indent = indent
            indent += ttcontainer.indent
            previous_indent2 = indent

            statement_selector_is_open = False
            num_children = len(ttcontainer)
            for i,child in enumerate(ttcontainer):
                if isinstance(child,tree.TTContainer):
                    if statement_selector_is_open:
                        indent = previous_indent2
                        trafo_info.generated_code += indent+"}\n"
                        traverse_(child)
                else:
                    if ( not statement_selector_is_open
                         and not resource_filter.worker_and_vector_partitioned_mode()
                         and not i == num_children-1 ):
                        trafo_info.generated_code += "{}if ({}) {{".format(
                          indent,
                          resource_filter.statement_selection_condition()
                        )
                        statement_selector_is_open = True
                        indent += opts.single_level_indent
                     traverse_(child)
            if statement_selector_is_open:
                trafo_info.generated_code += indent+"}\n"
            indent = previous_indent

        def traverse_(ttnode):
            nonlocal trafo_info
            nonlocal indent
            nonlocal resource_filter
            nonlocal ln_info
        
            lvalues, rvalues = [], []
            if isinstance(ttnode,IComputeConstruct):
                acc_construct_info = analysis.acc.analyze_directive(ttnode)  
                if acc_construct_info.is_serial:
                    trafo_info.max_num_gangs = "1"    
                    trafo_info.max_num_workers = "1"    
                    trafo_info.max_vector_length = "1"    
                    trafo_info.max_num_threads_per_gang = "1"
                else:
                    if acc_construct_info.num_gangs.specified:
                        trafo_info.max_num_gangs = acc_construct_info.num_gangs
                    if acc_construct_info.num_workers.specified:
                        trafo_info.max_num_workers = acc_construct_info.num_workers
                    if acc_construct_info.vector_length.specified:
                        trafo_info.max_vector_length = acc_construct_info.vector_length
                    trafo_info.max_num_workers = "1"    
                    trafo_info.max_vector_length = "1"    
                    trafo_info.max_num_threads_per_gang = "1"
                trafo_info.private_variables += ttnode.private_vars() 
                trafo_info.firstprivate_variables += ttnode.firstprivate_vars()

            elif isinstance(ttnode,TTDo):
                loopnest_open = ""
                loopnest_close = ""
                loopnest_resource_filter = loops.AccResourceFilter()
                loopnest_indent = ""
                #
                loopnest = ln_info.loopnest
                num_collapse = ln_info.num_collapse 
                tile_sizes = ln_info.tile_sizes
                if num_collapse > 1:
                    assert loopnest != none
                    loopnest.append(
                      _create_unpartitioned_loop(ttdo)
                    )      
                    if len(loopnest) == num_collapse:
                        loopnest_open,\
                        loopnest_close,\
                        loopnest_resource_filter,\
                        loopnest_indent = \
                            _render_loopnest(ln_info)
                        ln_info = None
                        num_collapse = 0   
                        # TODO check if we can move statement into inner loop, should be possible if 
                        # a loop statement is not relying on the lvalue or inout arg from a previous statement (difficult to analyze)
                        # alternative interpretation of collapse user information -> we can reorder statements without error
                elif len(tile_sizes) > 0:
                    assert loopnest != None
                    loopnest.append(
                      _create_unpartitioned_loop(ttdo)
                    )      
                    if len(loopnest) == len(tile_sizes):
                        loopnest_open,\
                        loopnest_close,\
                        loopnest_resource_filter,\
                        loopnest_indent = \
                          _render_loopnest(loopnest)
                        ln_info.reset()
                        tile_sizes = []
                    traverse_container_body_(ttnode)
                elif ttnode.annotation != None:
                    loop_annotation = ttnode.annotation
                    device_specs = loop_annotation.get_device_specs()
                    device_spec = next((d for d in device_specs if d.applies(device_type)),None)
                    assert device_spec != None
                    loop_parallelism = devicespec.parallelism()
                    _check_partitioning_mode(resource_filter,loop_parallelism)
                    loop_num_gangs,loop_num_workers,\
                    loop_vector_length,loop_num_threads_per_gang,\
                    ln_info.num_collapse, ln_info.tile_sizes =\
                        _get_specified_resources(device_specs,device_type)
                    ln_info.private_vars = loop_annotation.private_vars() 
                    loopnest = loops.Loopnest([
                      loops.Loop(
                        index = ttnode.loop_var(),
                        first = ttnode.begin_cstr(),
                        last = ttnode.end_cstr(),
                        step = ttnode.step_cstr() if ttnode.has_step() else None,
                        num_gangs = loop_num_gangs,
                        num_workers = loop_num_workers,
                        vector_length = loop_vector_length,
                        gang_partitioned = loop_annotation.gang_partitioned_mode(),
                        worker_partitioned = loop_annotation.worker_partitioned_mode(),
                        vector_partitioned = loop_annotation.vector_partitioned_mode()
                      )
                    ])
                    if ( num_collapse == 0
                         and not len(tile_sizes) ):
                        loopnest_open,\
                        loopnest_close,\
                        loopnest_resource_filter,\
                        loopnest_indent = \
                          _render_loopnest(loopnest)
                        ln_info.reset()
                    # loop directives might contain lvalues, rvalues that need to be passed
                    _find_lvalues_and_rvalues_in_directive(ttnode,lvalues,rvalues)
                else:
                    pass   
                resource_filter += loopnest_resource_filter
                previous_indent = indent
                indent += loopnest_indent
                #
                traverse_container_body_(ttnode)
                #
                resource_filter -= loopnest_resource_filter
                indent = previous_indent
            elif isinstance(ttnode,TTContainer):
                _find_lvalues_and_rvalues(ttnode,lvalues,rvalues)
                # TODO scan lvalue and rvalue
                trafo_info.generated_code += textwrap.indent(ttnode.header_cstr(),indent)
                indent += ttnode.indent
                #
                traverse_container_body_(ttnode)
                #
                trafo_info.generated_code += ttnode.footer_cstr()
            else: # other statements
                _find_lvalues_and_rvalues(ttnode,lvalues,rvalues)
                # TODO scan for lvalue and rvalues
                # TODO expand array assignment expressions
                # TODO modify expressions that are
                trafo_info.generated_code += ttnode.cstr().rstrip("\n")+"\n"
    traverse_(ttaccconstruct)
    return trafo_info.generated_code
    #preamble1 = []
    #preamble2 = []
    #indices = []
    #problem_sizes = []
    #indices.append("int _rem = __gidx1;\n") # linear index
    #indices.append("int _denom = _problem_size;\n")
    # 
    #preamble2.append("const int _problem_size = {};\n".format("*".join(problem_sizes)))
    #return preamble1+preamble2, indices, [ "__gidx1 < _problem_size" ]
