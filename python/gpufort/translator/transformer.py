import loop_transformations

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
    tile_sizes = None
    num_collapse = None   
 
    device_spec = next((d for d in device_specs if d.applies(device_type)),None)
    spec_num_gangs   = device_spec.num_gangs_teams_blocks() # list, if CUDA Fortran spec
    spec_num_workers = device_spec.num_workers() # no list
    spec_vector_length = device_spec.vector_length() # no list
    spec_num_threads_per_gang = device_spec.num_threads_in_block()
    if spec_num_gangs[0] not in [tree.grammar.CLAUSE_NOT_FOUND,
                                 tree.grammar.CLAUSE_VALUE_NOT_SPECIFIED]:
        num_gangs += [str(c) for c in spec_num_gangs]
    if spec_num_workers not in [tree.grammar.CLAUSE_NOT_FOUND,
                                tree.grammar.CLAUSE_VALUE_NOT_SPECIFIED]:
        num_workers = spec_num_workers
    if spec_vector_length not in [tree.grammar.CLAUSE_NOT_FOUND,
                                  tree.grammar.CLAUSE_VALUE_NOT_SPECIFIED]:
        vector_length = spec_vector_length
    if spec_num_threads_per_gang[0] not in [tree.grammar.CLAUSE_NOT_FOUND,
                                            tree.grammar.CLAUSE_VALUE_NOT_SPECIFIED]:
        num_threads_per_gang = spec_num_threads_per_gang
        # TODO implies:
        # vector_length = "warpSize"
        # num_workers = "*".join(num_threads_per_gang)/warpSize
        # num_gangs   = 

    return (num_gangs, num_workers, vector_length, num_threads_per_gang)

def transform(ttcomputeconstruct):
    results = [] # one per device spec
    device_types = _identify_device_types(ttcomputeconstruct)
    
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
    for device_type in device_types:
        indent = ""
        resource_filter = loop_transformations.ResourceFilter()
        loopnest = None
        #
        in_loop = False
        num_collapse = 0
        num_tile_dims = 0

        # for the whole compute construct
        max_num_gangs = [] # list
        max_num_threads_per_gang = [] # list
        max_num_workers = None
        max_vector_length = None
 
        # per loopnest in the compute construct
        all_num_gangs = []
        all_num_workers = []
        all_vector_length = []
        def traverse_(ttnode):
            nonlocal resource_filter
            nonlocal loopnest
            nonlocal in_loop
            nonlocal num_collapse
            nonlocal tile_sizes
            nonlocal max_num_gangs
            nonlocal max_num_threads_per_gang
            nonlocal max_num_workers
            nonlocal max_vector_length
            nonlocal all_num_gangs
            nonlocal all_num_workers
            nonlocal all_vector_length
            prev_resource_filter = resource_filter

            if isinstance(ttnode,IComputeConstruct):
                device_specs = ttnode.get_device_specs()
                max_num_gangs,max_num_workers,\
                max_vector_length,max_num_threads_per_gang,\
                _,_ =\
                    _get_specified_resources(device_specs,device_type)
            elif isinstance(ttnode,TTDo):
                loopnest_open  = ""
                loopnest_close = ""
                if num_collapse > 0:
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
                        resource_filter += loopnest_resource_filter
                        indent += loopnest_indent
                        loopnest = None
                        num_collapse = 0   
                    # TODO check if we can move statement into inner loop, should be possible if 
                    # a loop statement is not relying on the lvalue or inout arg from a previous statement (difficult to analyze)
                    # alternative interpretation of collapse user information -> we can reorder statements without error
                elif len(tile_sizes) > 0:
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
                        resource_filter += loopnest_resource_filter
                        indent += loopnest_indent
                        loopnest = None
                        tile_sizes = []
                elif ttnode.annotation != None:
                    # get device spec for type
                    device_specs = ttnode.annotation.get_device_specs()
                    device_spec = next((d for d in device_specs if d.applies(device_type)),None)
                    assert device_spec != None
                    loop_annotation = ttnode.annotation
                    if loop_annotation.parallelism.gang_parallelism():
                        if resource_filter.have_gang_parallelism():
                            raise util.eror.SyntaxError("already in gang-partitioned region")
                        elif resource_filter.have_worker_parallelism():
                            raise util.eror.SyntaxError("no gang partitioning possible in worker-partitioned region")
                        elif resource_filter.have_vector_parallelism():
                            raise util.eror.SyntaxError("no gang partitioning possible in vector-partitioned region")
                        # TODO get num gangs
                    if loop_annotation.parallelism().worker_parallelism():
                        if resource_filter.have_worker_parallelism():
                            raise util.eror.SyntaxError("already in worker-partitioned region")
                        elif resource_filter.have_vector_parallelism():
                            raise util.eror.SyntaxError("no worker partitioning possible in vector-partitioned region")
                    if loop_annotation.parallelism().vector_parallelism():
                        if resource_filter.have_vector_parallelism():
                            raise util.eror.SyntaxError("already in vector-partitioned region")
                    loop_num_gangs,loop_num_workers,\
                    loop_vector_length,loop_num_threads_per_gang,\
                    num_collapse, tile_sizes =\
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
                        gang_partitioned = loop_annotation.gang_parallelism(),
                        worker_partitioned = loop_annotation.worker_parallelism(),
                        vector_partitioned = loop_annotation.vector_parallelism()
                      )
                    ])
                    if ( num_collapse == 0
                         and not len(tile_sizes) ):
                        loopnest_open,\
                        loopnest_close,\
                        loopnest_resource_filter,\
                        loopnest_indent = \
                            loopnest.map_to_hip_cpp()
                        resource_filter += loopnest_resource_filter
                        indent += loopnest_indent
                        loopnest = None
                    # TODO transform according to parallelism_mode 
                    # consider that parallelism mode might not apply to finer-grain parallel-loops and further not 
                    # to conditional code around these loops
                    # need to check if a loop is parallelized
                    # get a statements parallelism level from a bottom-up search
                    # not all statements might be affected
                else:
                    pass   
            elif isinstance(ttnode,TTContainer):
                # traverse all elements in body
                pass
            else: # any other statement
                if ( resource_filter != prev_resource_filter ) {
                  any_resources_masked_out      = resource_filter. 
                  statement_selection_condition = resource_filter.statement_selection_condition()
                }
                pass

    #preamble1 = []
    #preamble2 = []
    #indices = []
    #problem_sizes = []
    #indices.append("int _rem = __gidx1;\n") # linear index
    #indices.append("int _denom = _problem_size;\n")
    # 


    #preamble2.append("const int _problem_size = {};\n".format("*".join(problem_sizes)))
    #return preamble1+preamble2, indices, [ "__gidx1 < _problem_size" ]
