import loop_transformations

from . import opts

loop_transformations.single_level_indent = opts.single_level_indent

def _identify_device_types(ttnode):
    device_types = set()
    directives = []
    def traverse_(ttnode):
        nonlocal device_types
        if isinstance(ttnode,(IComputeConstruct,ILoopAnnotation)):
            for device_spec in ttnode.get_device_specs():
                device_types.add(device_spec.device_type)
            for ttchild in ttnode:
                traverse_(ttchild)
    traverse_(ttnode)
    if not len(device_types):
        device_types.append("*")
    return device_types

def _get_specified_resources(device_specs,device_type):
    num_gangs = [] # list
    num_threads_per_gang = [] # list
    num_workers = None
    vector_length = None
    tile_sizes = []
    num_collapse = None   
 
    device_spec = next((d for d in device_specs if d.applies(device_type)),None)
    spec_num_gangs   = device_spec.num_gangs() # list, if CUDA Fortran spec
    spec_num_workers = device_spec.num_workers() # no list
    spec_vector_length = device_spec.vector_length() # no list
    spec_num_threads_per_gang = device_spec.num_threads_in_block()
    spec_num_collapse = device_spec.num_collapse()
    spec_tile_sizes = device_spec.tile_sizes()
    
    def is_specified_(arg):
        return arg not in [tree.grammar.CLAUSE_NOT_FOUND,
                           tree.grammar.CLAUSE_VALUE_NOT_SPECIFIED]

    if is_specified(spec_num_gangs[0]):
        num_gangs = [str(c) for c in spec_num_gangs]
    if is_specified_(spec_num_workers):
        num_workers = spec_num_workers
    if is_specified_(spec_vector_length):
        vector_length = spec_vector_length
    if is_specified_(spec_num_threads_per_gang[0]):
        num_threads_per_gang = spec_num_threads_per_gang
        # TODO implies:
        # vector_length = "warpSize"
        # num_workers = "*".join(num_threads_per_gang)/warpSize
        # num_gangs   = 
    if is_specified_(spec_num_collapse):
        num_collapse = spec_num_collapse
    if is_specified_(spec_tile_sizes[0]):
        tile_sizes = spec_tile_sizes
    return (num_gangs,
            num_workers,
            vector_length,
            num_threads_per_gang,
            num_collapse,
            tile_sizes)

class TransformationResult:
    def __init__(self):
        self.generated_code = ""
        self.max_num_gangs = [] # list
        self.max_num_threads_per_gang = [] # list
        self.max_num_workers = None
        self.max_vector_length = None
        self.all_num_gangs = []
        self.all_num_workers = []
        self.all_vector_length = []
        self.loop_variables = []
        self.gang_private_variables = []
        self.gang_firstprivate_variables = []
        self.gang_reductions = []
        self.other_variables = []
    def derive_grid_c_str():
        pass
    def derive_grid_f_str():
        pass
    def derive_block_size_c_str():
        pass
    def derive_block_size_f_str():
        pass

def LoopnestInfo:
    def __init__(self):
        self.loopnest = None
        self.num_collapse = 0
        self.tile_sizes = []
        self.private_vars = []
        self.first_private_vars = []
        self.reductions = []
        self.problem_sizes_f_str = []
        self.problem_sizes_c_str = []
    def reset(self):
        self.__init__()
 
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

def transform(ttcomputeconstruct):
    results = [] # one per device spec
    device_types = _identify_device_types(ttcomputeconstruct)
    
    for device_type in device_types:
        # for the whole compute construct
        transformation_result = TransformationResult() 
        resource_filter = loop_transformations.ResourceFilter()
        transformation_result.generated_code = ""
        indent = ""
        #
        # per loopnest in the compute construct
        loopnest_info = LoopnestInfo()
        def traverse_container_body_(ttcontainer):
            nonlocal transformation_result
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
                        transformation_result.generated_code += indent+"}\n"
                        traverse_(child)
                else:
                    if ( not statement_selector_is_open
                         and not resource_filter.worker_and_vector_partitioned_mode()
                         and not i == num_children-1 ):
                        transformation_result.generated_code += "{}if ({}) {{".format(
                          indent,
                          resource_filter.statement_selection_condition()
                        )
                        statement_selector_is_open = True
                        indent += opts.single_level_indent
                     traverse_(child)
            if statement_selector_is_open:
                transformation_result.generated_code += indent+"}\n"
            indent = previous_indent

        def traverse_(ttnode):
            nonlocal transformation_result
            nonlocal indent
            nonlocal resource_filter
            nonlocal loopnest_info

            if isinstance(ttnode,IComputeConstruct):
                device_specs = ttnode.get_device_specs()
                transformation_result.max_num_gangs,\
                transformation_result,max_num_workers,\
                transformation_result.max_vector_length,\
                transformation_result.max_num_threads_per_gang,\
                _,_ =\
                    _get_specified_resources(device_specs,device_type)
                transformation_result.private_variables = 
                transformation_result.firstprivate_variables = 
            elif isinstance(ttnode,TTDo):
                loopnest_open = ""
                loopnest_close = ""
                loopnest_resource_filter = loop_transformations.ResourceFilter()
                loopnest_indent = ""
                #
                loopnest = loopnest_info.loopnest
                num_collapse = loopnest_info.num_collapse 
                tile_sizes = loopnest_info.tile_sizes
                if num_collapse > 1:
                    assert loopnest != none
                    loopnest.append(
                      loop_transformations.Loop(
                        index = ttnode.loop_var(),
                        first = ttnode.begin_c_str(),
                        last = ttnode.end_c_str(),
                        step = ttnode.step_c_str() if ttnode.has_step() else None,
                      )
                    )      
                    if len(loopnest) == num_collapse:
                        loopnest_open,\
                        loopnest_close,\
                        loopnest_resource_filter,\
                        loopnest_indent = \
                            loopnest.collapse().map_to_hip_cpp()
                        loopnest_info = None
                        num_collapse = 0   
                    # TODO check if we can move statement into inner loop, should be possible if 
                    # a loop statement is not relying on the lvalue or inout arg from a previous statement (difficult to analyze)
                    # alternative interpretation of collapse user information -> we can reorder statements without error
                elif len(tile_sizes) > 0:
                    assert loopnest != None
                    loopnest.append(
                      loop_transformations.Loop(
                        index = ttnode.loop_var(),
                        first = ttnode.begin_c_str(),
                        last = ttnode.end_c_str(),
                        step = ttnode.step_c_str() if ttnode.has_step() else None,
                      )
                    )
                    if len(loopnest) == len(tile_sizes):
                        loopnest_open,\
                        loopnest_close,\
                        loopnest_resource_filter,\
                        loopnest_indent = \
                            loopnest.tile(tile_sizes).map_to_hip_cpp()
                        #resource_filter += loopnest_resource_filter
                        #indent += loopnest_indent
                        loopnest_info.reset()
                        tile_sizes = []
                    traverse_container_body_(ttnode)
                elif ttnode.annotation != None:
                    # get device spec for type
                    loop_annotation = ttnode.annotation
                    device_specs = loop_annotation.get_device_specs()
                    device_spec = next((d for d in device_specs if d.applies(device_type)),None)
                    assert device_spec != None
                    loop_parallelism = devicespec.parallelism()
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
                    loop_num_gangs,loop_num_workers,\
                    loop_vector_length,loop_num_threads_per_gang,\
                    loopnest_info.num_collapse, loopnest_info.tile_sizes =\
                        _get_specified_resources(device_specs,device_type)
                    loopnest = loop_transformations.Loopnest([
                      loop_transformations.Loop(
                        index = ttnode.loop_var(),
                        first = ttnode.begin_c_str(),
                        last = ttnode.end_c_str(),
                        step = ttnode.step_c_str() if ttnode.has_step() else None,
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
                            loopnest.map_to_hip_cpp()
                        #resource_filter += loopnest_resource_filter
                        #indent += loopnest_indent
                        loopnest = None
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
                # TODO scan lvalue and rvalue
                transformation_result.generated_code += textwrap.indent(ttnode.header_c_str(),indent)
                indent += ttnode.indent
                #
                traverse_container_body_(ttnode)
                #
                transformation_result.generated_code += ttnode.footer_c_str()
            else: # other statements
                transformation_result.generated_code += ttnode.c_str().rstrip("\n")+"\n"
    traverse_(ttcomputeconstruct)
    return transformation_result.generated_code
    #preamble1 = []
    #preamble2 = []
    #indices = []
    #problem_sizes = []
    #indices.append("int _rem = __gidx1;\n") # linear index
    #indices.append("int _denom = _problem_size;\n")
    # 
    #preamble2.append("const int _problem_size = {};\n".format("*".join(problem_sizes)))
    #return preamble1+preamble2, indices, [ "__gidx1 < _problem_size" ]
