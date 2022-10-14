# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import indexer

from .. import tree

from . import loopmgr

def _collect_ranges(function_call_args,include_none_values=False):
    ttranges = []
    for i,ttnode in enumerate(function_call_args):
        if isinstance(ttnode, tree.TTSlice):
            ttranges.append(ttnode)
        elif include_none_values:
            ttranges.append(None)
    return ttranges

def _collect_ranges_in_ttvalue(ttvalue,include_none_values=False):
    """
    :return A list of range objects. If the list is empty, no function call
            or tensor access has been found. If the list contains a None element
            this implies that a function call or tensor access has been found 
            but a scalar index argument was used.
    """
    current = ttvalue._value
    if isinstance(current,tree.TTTensorEval):
        return _collect_ranges(current._args,include_none_values)
    elif isinstance(current,tree.TTDerivedTypeMember):
        result = []
        while isinstance(current,tree.TTDerivedTypeMember):
            if isinstance(current._type,tree.TTTensorEval):
                result += _collect_ranges(current._type._args,include_none_values)
            if isinstance(current._element,tree.TTTensorEval):
                result += _collect_ranges(current._element._args,include_none_values)
            current = current._element
        return result
    else:
        return []

def is_array_assignment(ttlvalue,scope):
    """
    :return: If the LHS expression is an identifier and the associated variable
             is an array or if the LHS expression is an array colon expression 
             such as `A(:,1:n)`.
    """
    lvalue_fstr = ttlvalue.fstr()
    livar = indexer.scope.search_scope_for_var(scope,lvalue_fstr)
    loop_indices = [] 
    if livar["rank"] > 0:
        lvalue_ranges_or_none = _collect_ranges_in_ttvalue(ttlvalue,include_none_values=True)
        lvalue_ranges = [r for r in lvalue_ranges_or_none if r != None]
        if len(lvalue_ranges_or_none):
            num_implicit_loops = len(lvalue_ranges)
        else:
            num_implicit_loops = livar["rank"]
    return num_implicit_loops > 0

def is_array_reduction_intrinsic_call(ttassignment,scope):
    """
    x = OP(ARRAY,args)
    where OP is one of
    SUM(SOURCE[,DIM][,MASK]) -- sum of array elements (in an optionally specified dimension under an optional mask).
    MAXVAL(SOURCE[,DIM][,MASK]) -- maximum Value in an array (in an optionally specified dimension
                                   under an optional mask);
    MINVAL(SOURCE[,DIM][,MASK]) -- minimum value in an array (in an optionally specified dimension
                                  under an optional mask); 
    ALL(MASK[,DIM]) -- .TRUE. if all values are .TRUE., (in an optionally specified dimension);
    ANY(MASK[,DIM]) -- .TRUE. if any values are .TRUE., (in an optionally specified dimension);
    COUNT(MASK[,DIM]) -- number of .TRUE. elements in an array, (in an optionally specified dimension);
    """
    reduction_ops = {}
    #value_type, index_record = indexer.scope.search_scope_for_value_expr(scope, ident)
    #if value_type == indexer.indexertypes.ValueType.VARIABLE:
    #    value._value._type = tree.TTTensorEval.Type.ARRAY_ACCESS
    #elif value_type == indexer.indexertypes.ValueType.PROCEDURE:
    #    value._value._type = tree.TTTensorEval.Type.FUNCTION_CALL
    #elif value_type == indexer.indexertypes.ValueType.INTRINSIC:
    #    value._value._type = tree.TTTensorEval.Type.INTRINSIC_CALL
     
    return False    

def is_array_transformation_intrinsic_call(ttassignment,scope):
    """
    """
    return False    

def expand(ttassignment,scope):
    loopmgr = None
    body = None 
    return (loopmgr, body)

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
    lvalue_fstr = ttlvalue.fstr()
    livar = indexer.scope.search_scope_for_var(scope,lvalue_fstr)
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
                    lvalue_ranges[i].overwrite_fstr(idx)
            else:
                ttlvalue.overwrite_fstr(
                        "".join([ttlvalue.identifier_part(tree.make_fstr),
                            "(",",".join(loop_indices),")"]))
            for ttrvalue in tree.find_all(ttassignment._rhs, searched_type=tree.TTRvalue):
                try:
                    rvalue_fstr = ttrvalue.fstr()
                    rivar = indexer.scope.search_scope_for_var(scope,rvalue_fstr)
                    if rivar["rank"] > 0:
                        rvalue_ranges_or_none = _collect_ranges_in_ttvalue(ttrvalue,include_none_values=True)
                        rvalue_ranges = [r for r in rvalue_ranges_or_none if r != None] 
                        if len(rvalue_ranges) == num_implicit_loops:
                            for i,idx in enumerate(loop_indices):
                                rvalue_ranges[i].overwrite_fstr(idx)
                        elif not len(rvalue_ranges_or_none) and rivar["rank"] == num_implicit_loops:
                            ttrvalue.overwrite_fstr(
                                    "".join([ttrvalue.identifier_part(tree.make_fstr),
                                        "(",",".join(loop_indices),")"])) 
                        elif len(rvalue_ranges) or rivar["rank"] != num_implicit_loops:
                            raise util.error.LimitationError("failed to expand colon operator expression to loopnest: not enough colon expressions in rvalue argument list")
                except util.error.LookupError:
                    pass
            f_expr = ttassignment._lhs.identifier_part(tree.make_fstr)
            do_loop_statements, end_do_statements = _create_do_loop_statements(f_expr,lvalue_ranges,loop_indices,fortran_style_tensors)
            statements = do_loop_statements + [ttassignment.fstr()] + end_do_statements
            return statements, int_counter + len(loop_indices), True
        else:
            return [], int_counter, False
    else:
        return [], int_counter, False