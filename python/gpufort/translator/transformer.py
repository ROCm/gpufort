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
    tile_sizes = []
    num_collapse = None   
 
    device_spec = next((d for d in device_specs if d.applies(device_type)),None)
    spec_num_gangs   = device_spec.num_gangs_teams_blocks() # list, if CUDA Fortran spec
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

def transform(ttcomputeconstruct,
              gang_partitioned = False,
              worker_partitioned = False,
              vector_partitioned = False):
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
        generated_code = ""
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

        def traverse_container_body_(ttcontainer):
            nonlocal generated_code
            nonlocal indent
            nonlocal resource_filter

            statement_selector_is_open = False
            num_children = len(ttcontainer)
            for i,child in enumerate(ttcontainer):
                if isinstance(child,tree.TTContainer):
                    if statement_selector_is_open:
                        generated_code += indent+"}\n"
                        traverse_(child)
                else:
                    if ( not statement_selector_is_open
                         and not resource_filter.worker_and_vector_partitioned_mode()
                         and not i == num_children-1 ):
                        generated_code += "{}if ({}) {{".format(
                          indent,
                          resource_filter.statement_selection_condition()
                        )
                        statement_selector_is_open = True
                        traverse_(child)

        def traverse_(ttnode):
            nonlocal generated_code
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

            if isinstance(ttnode,IComputeConstruct):
                device_specs = ttnode.get_device_specs()
                max_num_gangs,max_num_workers,\
                max_vector_length,max_num_threads_per_gang,\
                _,_ =\
                    _get_specified_resources(device_specs,device_type)
            elif isinstance(ttnode,TTDo):
                loopnest_open  = ""
                loopnest_close = ""
                loopnest_resource_filter = loop_transformations.ResourceFilter()
                loopnest_indent = ""
                if num_collapse > 0:
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
                        #resource_filter += loopnest_resource_filter
                        #indent += loopnest_indent
                        loopnest = None
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
                        loopnest = None
                        tile_sizes = []
                    traverse_container_body_(ttnode)
                elif ttnode.annotation != None:
                    # get device spec for type
                    device_specs = ttnode.annotation.get_device_specs()
                    device_spec = next((d for d in device_specs if d.applies(device_type)),None)
                    assert device_spec != None
                    loop_annotation = ttnode.annotation
                    if loop_annotation.parallelism.gang_partitioned_mode():
                        if resource_filter.gang_partitioned_mode():
                            raise util.eror.SyntaxError("already in gang-partitioned region")
                        elif resource_filter.worker_partitioned_mode():
                            raise util.eror.SyntaxError("no gang partitioning possible in worker-partitioned region")
                        elif resource_filter.vector_partitioned_mode():
                            raise util.eror.SyntaxError("no gang partitioning possible in vector-partitioned region")
                        # TODO get num gangs
                    if loop_annotation.parallelism().worker_partitioned_mode():
                        if resource_filter.worker_partitioned_mode():
                            raise util.eror.SyntaxError("already in worker-partitioned region")
                        elif resource_filter.vector_partitioned_mode():
                            raise util.eror.SyntaxError("no worker partitioning possible in vector-partitioned region")
                    if loop_annotation.parallelism().vector_partitioned_mode():
                        if resource_filter.vector_partitioned_mode():
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
                    # TODO transform according to parallelism_mode 
                    # consider that parallelism mode might not apply to finer-grain parallel-loops and further not 
                    # to conditional code around these loops
                    # need to check if a loop is parallelized
                    # get a statements parallelism level from a bottom-up search
                    # not all statements might be affected
                else:
                    pass   
                resource_filter += loopnest_resource_filter
                previous_indent = indent
                indent += loopnest_indent
                traverse_container_body_(ttnode)
                resource_filter -= loopnest_resource_filter
                indent = previous_indent
            elif isinstance(ttnode,TTContainer): 
                generated_code += 
                traverse_container_body_(ttnode)
                
            else: # other statements
                

    #preamble1 = []
    #preamble2 = []
    #indices = []
    #problem_sizes = []
    #indices.append("int _rem = __gidx1;\n") # linear index
    #indices.append("int _denom = _problem_size;\n")
    # 


    #preamble2.append("const int _problem_size = {};\n".format("*".join(problem_sizes)))
    #return preamble1+preamble2, indices, [ "__gidx1 < _problem_size" ]
