# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import util
from gpufort import indexer

from . import tree
from . import analysis
from . import parser
from . import opts
from . import prepostprocess

def _collect_ranges(function_call_args,include_none_values=False):
    ttranges = []
    for i,ttnode in enumerate(function_call_args):
        if isinstance(ttnode, tree.TTRange):
            ttranges.append(ttnode)
        elif include_none_values:
            ttranges.append(None)
    return ttranges

def _collect_ranges_in_ttvalue(ttvalue,include_none_values=False):
    """
    :return A list of range objects. If the list is empty, no function call
            or tensor access has been found. If the list contains a None element
            this implies that a function call or tensor access has been found 
            was scalar index argument was used.
    """
    current = ttvalue._value
    if isinstance(current,tree.TTFunctionCallOrTensorAccess):
        return _collect_ranges(current._args,include_none_values)
    elif isinstance(current,tree.TTDerivedTypeMember):
        result = []
        while isinstance(current,tree.TTDerivedTypeMember):
            if isinstance(current._type,tree.TTFunctionCallOrTensorAccess):
                result += _collect_ranges(current._type._args,include_none_values)
            if isinstance(current._element,tree.TTFunctionCallOrTensorAccess):
                result += _collect_ranges(current._element._args,include_none_values)
            current = current._element
        return result
    else:
        return []

def _create_do_loop_statements(name,ranges,loop_indices,fortran_style_tensors):
    do_statements     = []
    end_do_statements = []
    for i, loop_idx in enumerate(loop_indices,1):
        if len(ranges):
            ttrange = ranges[i-1]
            lbound = ttrange.l_bound(tree.make_f_str)
            ubound = ttrange.u_bound(tree.make_f_str)
            stride = ttrange.stride()
        else:
            lbound = ""
            ubound = ""
            stride = "1"
        if not len(lbound):
            if fortran_style_tensors:
                lbound = "lbound({name},{i})".format(name=name, i=i)
            else:
                lbound = "{name}_lb{i}".format(name=name, i=i)
        if not len(ubound):
            if fortran_style_tensors:
                ubound = "ubound({name},{i})".format(name=name, i=i)
            else:
                ubound = "({ubound} + {name}_n{i} - 1)".format(ubound=ubound,
                                                               name=name,
                                                               i=i)
        if len(stride):
            stride = "," + stride
        do_statements.insert(0,"do {var}={lb},{ub}{stride}".format(
            var=loop_idx, lb=lbound, ub=ubound,
            stride=stride))
        end_do_statements.insert(0,"end do")
    return do_statements, end_do_statements

def _expand_array_expression(ttassignment,scope,int_counter,fortran_style_tensors):
    """Expand expressions such as 
    `a = b`
    or 
    `a(:) = b(1,1:n)`,
    where a and b are arrays, to a nest of do - loops
    that sets the individual elements.

    :note: Currently, the code generator only if number of array
    range expressions is equal for rvalues and lvalue.
    No further checks are performed, e.g. if function calls
    are present in the expression.
    """
    ttlvalue = ttassignment._lhs
    lvalue_f_str = ttlvalue.f_str()
    livar = indexer.scope.search_scope_for_var(scope,lvalue_f_str)
    loop_indices = [] 
    if livar["rank"] > 0:
        lvalue_ranges_or_none = _collect_ranges_in_ttvalue(ttlvalue,include_none_values=True)
        lvalue_ranges = [r for r in lvalue_ranges_or_none if r != None]
        if len(lvalue_ranges_or_none):
            num_implicit_loops = len(lvalue_ranges)
        else:
            num_implicit_loops = livar["rank"]
        if num_implicit_loops > 0:
            loop_indices = [ "".join(["_i",str(int_counter+i)]) for i in range(0,num_implicit_loops) ]
            if len(lvalue_ranges):
                for i,idx in enumerate(loop_indices):
                    lvalue_ranges[i].overwrite_f_str(idx)
            else:
                ttlvalue.overwrite_f_str(
                        "".join([ttlvalue.identifier_part(tree.make_f_str),
                            "(",",".join(loop_indices),")"]))
            for ttrvalue in tree.find_all(ttassignment._rhs, searched_type=tree.TTRValue):
                try:
                    rvalue_f_str = ttrvalue.f_str()
                    rivar = indexer.scope.search_scope_for_var(scope,rvalue_f_str)
                    if rivar["rank"] > 0:
                        rvalue_ranges_or_none = _collect_ranges_in_ttvalue(ttrvalue,include_none_values=True)
                        rvalue_ranges = [r for r in rvalue_ranges_or_none if r != None] 
                        if len(rvalue_ranges) == num_implicit_loops:
                            for i,idx in enumerate(loop_indices):
                                rvalue_ranges[i].overwrite_f_str(idx)
                        elif not len(rvalue_ranges_or_none) and rivar["rank"] == num_implicit_loops:
                            ttrvalue.overwrite_f_str(
                                    "".join([ttrvalue.identifier_part(tree.make_f_str),
                                        "(",",".join(loop_indices),")"])) 
                        elif len(rvalue_ranges) or rivar["rank"] != num_implicit_loops:
                            raise util.error.LimitationError("failed to expand colon operator expression to loopnest: not enough colon expressions in rvalue argument list")
                except util.error.LookupError:
                    pass
            f_expr = ttassignment._lhs.identifier_part(tree.make_f_str)
            do_loop_statements, end_do_statements = _create_do_loop_statements(f_expr,lvalue_ranges,loop_indices,fortran_style_tensors)
            statements = do_loop_statements + [ttassignment.f_str()] + end_do_statements
            return statements, int_counter + len(loop_indices), True
        else:
            return [], int_counter, False
    else:
        return [], int_counter, False
    
def _traverse_tree(current,idx,parent,callback):
    callback(current,idx,parent)
    if isinstance(current,tree.TTContainer):
        for i,child in enumerate(current):
            _traverse_tree(child,i,current,callback)

def expand_all_array_expressions(ttcontainer,scope,fortran_style_tensors=True):
    int_counter = 1
    def callback_(current,idx,parent):
        nonlocal int_counter
        if isinstance(current,tree.TTAssignment):
            statements, int_counter, modified =\
              _expand_array_expression(current,scope,int_counter,
                                       fortran_style_tensors)
            if modified:
                ttdo = parser.parse_fortran_code(statements).body[0] # parse returns ttroot
                parent.body[idx] = ttdo
    assert isinstance(ttcontainer,tree.TTContainer)
    _traverse_tree(ttcontainer,0,ttcontainer.parent,callback_)
    return int_counter

def adjust_explicitly_mapped_arrays_in_rank(ttvalues,explicitly_mapped_vars):
    """
    :note: Must be applied after array expression expansion has
           completed.
    :note: Must be ensured that all value types with arguments are 
           flagged as tensor or not. 
    """
    c_ranks = {}
    for ttvalue in ttvalues:
        value_tag  = indexer.scope.create_index_search_tag_for_var(ttvalue.f_str())
        for var_expr in explicitly_mapped_vars:
            mapping_tag  = indexer.scope.create_index_search_tag_for_var(var_expr)
            if value_tag == mapping_tag:
                var_ttvalue = tree.grammar.lvalue.parseString(var_expr,parseAll=True)[0] # TODO analyse usage and directly return as type?
                if len(var_ttvalue.range_args()) < len(var_ttvalue.args()): # implies there is a fixed dimension
                    assert ttvalue.has_args()
                    if len(var_ttvalue.args()) > len(ttvalue.args()):
                        raise util.error.SyntaxError("Explicitly mapped expression has higher rank than actual variable")
                    ttvalue.args().max_rank   = len(var_ttvalue.range_args())
                    c_ranks[value_tag] = ttvalue.args().max_rank
    return c_ranks

def move_statements_into_compute_construct_body(ttcomputeconstruct):
    # TODO
    # subsequent loop ranges must not depend on LHS of assignment
    # or inout, out arguments of function call
    pass 

def _loop_range_c_str(ttdo,counter):
    result = [
      "_begin{} = {}".format(counter,ttdo.begin_c_str()),
      "_end{} = {}".format(counter,ttdo.end_c_str()),
    ]
    if ttdo.has_step():
        result.append("_step{} = {}".format(counter,ttdo.step_c_str()))
    else:
        result.append("_step{} = 1".format(counter))
    return "const int " + ",".join(result) + ";\n"

def _collapsed_loop_index_c_str(ttdo,counter):
    idx = ttdo.loop_var()
    args = [
      ttdo.thread_index,
      "_begin{}".format(counter),
      "_len{}".format(counter),
      "_step{}".format(counter),
    ]
    return "int {idx} = outermost_index_w_len({args});\n".format(\
           idx=idx,args=",".join(args))

def __identify_device_types(ttnode):
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
    
class ParallelismMode(enum.Enum):
    UNKNOWN=-1
    REDUNDANT=0
    SINGLE=1
    PARTITIONED=2

def transform(ttcomputeconstruct):

    results = [] # one per device spec
    device_types = __identify_device_types(ttcomputeconstruct)
    
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
        in_loop = False
        
        parallelism_level = [directives.Parallelism.GANG]
        parallelism_mode  = [ParallelismMode.REDUNDANT]
        num_collapse = 0
        num_tile_dims = 0
        # for the whole compute construct
        max_num_gangs = None
        max_num_workers = None
        max_vector_length = None
        # per loop nest in the compute construct
        num_gangs     = []
        num_workers   = []
        vector_length = []
        def traverse_(ttnode,num_collapse,mapped_loops,device_type):
            if isinstance(ttnode,IComputeConstruct):
                device_specs = ttnode.get_device_specs()
                device_spec = next((d for d in device_specs if d.applies(device_type)),None)
                assert device_spec != None
                cc_num_gangs   = ttnode.num_gangs_teams_blocks() # list
                cc_num_workers = ttnode.num_workers() # no list
                if cc_num_gangs[0] != tree.grammar.CLAUSE_NOT_FOUND:
                    max_num_gangs = "*".join(["("+str(g)+")" for g in cc_num_gangs]) 
                if cc_num_workers != tree.grammar.CLAUSE_NOT_FOUND:
                    max_num_workers = str(cc_num_workers)
            elif isinstance(ttnode,TTDo):
                if num_collapse > 0:
                    # TODO check iif we can move statement into inner loop, should be possible if 
                    # a loop statement is not relying on the lvalue or inout arg from a previous statement (difficult to analyze)
                    # alternative interpretation of collapse user information -> we can reorder statements without error
                    # Problem size should be 
                elif num_tile_dims > 0:
                    # gang parallism level
                    # append to grid size
                    # append to block size 
                elif ttnode.annotation != None:
                    # get device spec for type
                    device_specs = ttnode.annotation.get_device_specs()
                    device_spec = next((d for d in device_specs if d.applies(device_type)),None)
                    assert device_spec != None
                    if parallelism_level[-1] == directives.Parallelism.GANG:
                        if ttnode.annotation.parallelism() == directives.Parallelism.GANG:
                            parallelism_mode[-1] = Parallelism.PARTITIONED
                            #ttnode.partition_among_gangs()
                            pass
                        elif ttnode.annotation.parallelism() == directives.Parallelism.WORKER:
                            parallelism_level.append(Parallelism.WORKER)
                            parallelism_mode.append(Parallelism.PARTITIONED)
                            #ttnode.partition_among_gang_workers()
                            pass
                            #descend
                        elif ttnode.annotation.parallelism() == directives.Parallelism.VECTOR:
                            parallelism_level.append(Parallelism.VECTOR)
                            parallelism_mode.append(Parallelism.PARTITIONED)
                            #ttnode.partition_among_gang_workers()
                            pass
                            #descend
                        elif ttnode.annotation.parallelism() == directives.Parallelism.GANG_WORKER:
                            parallelism_level[-1] = Parallelism.GANG_WORKER
                            parallelism_mode[-1] = Parallelism.PARTITIONED
                            #ttnode.partition_among_all_workers()
                            pass
                            #descend
                        elif ttnode.annotation.parallelism() == directives.Parallelism.GANG_VECTOR:
                            parallelism_level[-1] = Parallelism.GANG_VECTOR
                            parallelism_mode[-1] = Parallelism.PARTITIONED
                            #ttnode.partition_among_all_threads()
                            pass
                            #descend
                        else:
                            parallelism_level.append(Parallelism.SEQ)
                            parallelism_mode.append(ParallelismMode.UNKNOWN)
                            #descend
                    elif parallelism_level[-1] == directives.Parallelism.WORKER:
                        if ttnode.annotation.parallelism() in [ directives.Parallelism.GANG,
                                                                directives.Parallelism.GANG_WORKER,
                                                                directives.Parallelism.GANG_VECTOR ]:
                            raise util.error.SyntaxError("no gang parallelism possible in worker parallelism section")
                        elif ttnode.annotation.parallelism() == directives.Parallelism.WORKER:
                            parallelism_mode[-1] == Parallelism.PARTITIONED)
                            #ttnode.partition_among_gang_workers()
                            pass
                        elif ttnode.annotation.parallelism() == directives.Parallelism.VECTOR:
                            parallelism_level.append(Parallelism.VECTOR)
                            parallelism_mode.append(Parallelism.PARTITIONED)
                            #ttnode.partition_among_gang_workers()
                            pass
                    elif parallelism_level[-1] == directives.Parallelism.WORKER:
                        if ttnode.annotation.parallelism() in [ directives.Parallelism.GANG,
                                                                directives.Parallelism.GANG_WORKER,
                                                                directives.Parallelism.GANG_VECTOR,
                                                                directives.Parallelism.WORKER,
                                                                directives.Parallelism.WORKER_VECTOR ]:
                            raise util.error.SyntaxError("no worker or gang parallelism possible in vector parallelism section")
                        elif ttnode.annotation.parallelism() == directives.Parallelism.VECTOR:
                            parallelism_level.append(Parallelism.VECTOR)
                            parallelism_mode.append(Parallelism.PARTITIONED)
                            #ttnode.partition_among_vector_lanes()
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

# TODO make use of parallelism-level here
def collapse_loopnest(ttdos):
    # TODO Traverse the whole tree and modify all TTCycle and Exit statements 
    preamble1 = []
    preamble2 = []
    indices = []
    problem_sizes = []
    indices.append("int _rem = __gidx1;\n")
    indices.append("int _denom = _problem_size;\n")
    for i,ttdo in enumerate(ttdos,1):
        ttdo.thread_index = "_rem,_denom" # side effects
        preamble1.append(_loop_range_c_str(ttdo,i))
        preamble2.append("const int _len{0} = loop_len(_begin{0},_end{0},_step{0});\n".format(i))
        problem_sizes.append("_len{}".format(i))
        for child in ttdo:
            if isinstance(child,tree.TTCycle):
                child._in_loop = False
            if isinstance(child,tree.TTExit):
                child._in_loop = False
        indices.append(_collapsed_loop_index_c_str(ttdo,i))
    # conditions = [ ttdos[0].hip_thread_bound_c_str() ]
    preamble2.append("const int _problem_size = {};\n".format("*".join(problem_sizes)))
    return preamble1+preamble2, indices, [ "__gidx1 < _problem_size" ]

def map_compute_construct_to_grid(ttdos):
    thread_indices = ["x", "y", "z"]
    while len(thread_indices) > len(ttdos):
        thread_indices.pop()
    indices = []
    conditions = []
    for ttdo in ttdos:
        ttdo.thread_index = thread_indices.pop()
        indices.append(ttdo.hip_thread_index_c_str())
        conditions.append(ttdo.hip_thread_bound_c_str())
        if not len(thread_indices):
            break
    return indices, conditions

def map_allocatable_pointer_derived_type_members_to_flat_arrays(ttvalues,loop_vars,scope):
    r"""Converts derived type expressions whose innermost element is an array of allocatable or pointer
    type to flat arrays in expectation that these will be provided
    to kernel as flat arrays too.
    :return: A dictionary containing substitutions for the identifier part of the 
             original variable expressions.
    """
    substitutions = {}
    for ttvalue in ttvalues:
        ttnode = ttvalue.get_value()
        if isinstance(ttnode,tree.TTDerivedTypeMember):
            ident = ttnode.identifier_part()
            ivar = indexer.scope.search_scope_for_var(scope,ident)
            if (ivar["rank"] > 0
                and ("allocatable" in ivar["attributes"]
                    or "pointer" in ivar["attributes"])):
                # TODO 
                # search through the subtree and ensure that only
                # the last element is an array indexed by the loop
                # variables 
                # Deep copy required for expressions that do not match
                # this criterion
                if (":" in ident 
                   or "(" in ident): # TODO hack
                    raise util.error.LimitationError("cannot map expression '{}'".format(ident))
                var_expr = indexer.scope.create_index_search_tag_for_var(ident)
                c_name = util.parsing.mangle_fortran_var_expr(var_expr) 
                substitute = ttnode.f_str().replace(ident,c_name)
                ttnode.overwrite_c_str(substitute)
                substitutions[var_expr] = c_name
    return substitutions

def map_scalar_derived_type_members_to_flat_scalars(ttvalues,loop_vars,scope):
    r"""Converts derived type expressions whose innermost element is a basic
    scalar type to flat arrays in expectation that these will be provided
    to kernel as flat scalars too via first private.
    :return: A dictionary containing substitutions for the identifier part of the 
             original variable expressions.
    """
    substitutions = {}
    for ttvalue in ttvalues:
        ttnode = ttvalue.get_value()
        if isinstance(ttnode,tree.TTDerivedTypeMember):
            ident = ttnode.identifier_part()
            ivar = indexer.scope.search_scope_for_var(scope,ident)
            if (ivar["rank"] == 0
               and ivar["f_type"] != "type"):
                # TODO 
                # search through the subtree and ensure that only
                # the last element is an array indexed by the loop
                # variables 
                # Deep copy required for expressions that do not match
                # this criterion
                if (":" in ident 
                   or "(" in ident): # TODO hack
                    raise util.error.LimitationError("cannot map expression '{}'".format(ident))
                var_expr = indexer.scope.create_index_search_tag_for_var(ident)
                c_name = util.parsing.mangle_fortran_var_expr(var_expr) 
                substitute = ttnode.f_str().replace(ident,c_name)
                ttnode.overwrite_c_str(substitute)
                substitutions[var_expr] = c_name
            elif (ivar["rank"] == 0
               and ivar["f_type"] == "type"):
                raise util.error.LimitationError("cannot map derived type members of derived type '{}'".format(ident))
    return substitutions

def flag_tensors(ttvalues, scope):
    """Clarify types of function calls / tensor access that are not members of a struct.
    Lookup order: Explicitly types variables -> Functions -> Intrinsics -> Implicitly typed variables
    """
    for value in ttvalues:
        ident = value.identifier_part()
        if isinstance(value._value, tree.TTFunctionCallOrTensorAccess):
            if ident.startswith("_"):
                # function introduced by GPUFORT, Fortran identifiers never start with '_'
                value._value._type = tree.TTFunctionCallOrTensorAccess.Type.FUNCTION_CALL
            else:
                try:
                   _ = indexer.scope.search_scope_for_var(scope, ident, 
                           consider_implicit = False) # just check if the var exists
                   value._value._type = tree.TTFunctionCallOrTensorAccess.Type.ARRAY_ACCESS
                except util.error.LookupError:
                   try:
                       _ = indexer.scope.search_scope_for_procedure(scope, ident) # just check if the procedure exists
                       value._value._type = tree.TTFunctionCallOrTensorAccess.Type.FUNCTION_CALL
                   except util.error.LookupError:
                       if indexer.scope.is_intrinsic(ident):
                          value._value._type = tree.TTFunctionCallOrTensorAccess.Type.INTRINSIC_CALL
                       else:
                           # TODO check EXTERNAL procedures too 
                           raise util.error.LookupError("expression '"+ident+"' could not be associated with any variable, procedure, or intrinsic")
