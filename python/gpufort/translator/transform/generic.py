# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import util
from gpufort import indexer

from .. import tree
from .. import analysis
from .. import parser
from .. import opts
from .. import prepostprocess

from . import loops

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
            was scalar index argument was used.
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

def _create_do_loop_statements(name,ranges,loop_indices,fortran_style_tensors):
    do_statements     = []
    end_do_statements = []
    for i, loop_idx in enumerate(loop_indices,1):
        if len(ranges):
            ttrange = ranges[i-1]
            lbound = ttrange.l_bound(tree.make_fstr)
            ubound = ttrange.u_bound(tree.make_fstr)
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
        value_tag  = indexer.scope.create_index_search_tag_for_var(ttvalue.fstr())
        for var_expr in explicitly_mapped_vars:
            mapping_tag  = indexer.scope.create_index_search_tag_for_var(var_expr)
            if value_tag == mapping_tag:
                var_ttvalue = tree.grammar.lvalue.parseString(var_expr,parseAll=True)[0] # todo: analyse usage and directly return as type?
                if len(var_ttvalue.slice_args()) < len(var_ttvalue.args()): # implies there is a fixed dimension
                    assert ttvalue.has_args()
                    if len(var_ttvalue.args()) > len(ttvalue.args()):
                        raise util.error.SyntaxError("Explicitly mapped expression has higher rank than actual variable")
                    ttvalue.args().max_rank   = len(var_ttvalue.slice_args())
                    c_ranks[value_tag] = ttvalue.args().max_rank
    return c_ranks

def _loop_range_cstr(ttdo,counter):
    result = [
      "_begin{} = {}".format(counter,ttdo.begin_cstr()),
      "_end{} = {}".format(counter,ttdo.end_cstr()),
    ]
    if ttdo.has_step():
        result.append("_step{} = {}".format(counter,ttdo.step_cstr()))
    else:
        result.append("_step{} = 1".format(counter))
    return "const int " + ",".join(result) + ";\n"

def _collapsed_loop_index_cstr(ttdo,counter):
    idx = ttdo.loop_var()
    args = [
      ttdo.thread_index,
      "_begin{}".format(counter),
      "_len{}".format(counter),
      "_step{}".format(counter),
    ]
    return "int {idx} = outermost_index_w_len({args});\n".format(\
           idx=idx,args=",".join(args))

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
                # todo: 
                # search through the subtree and ensure that only
                # the last element is an array indexed by the loop
                # variables 
                # Deep copy required for expressions that do not match
                # this criterion
                if (":" in ident 
                   or "(" in ident): # todo: hack
                    raise util.error.LimitationError("cannot map expression '{}'".format(ident))
                var_expr = indexer.scope.create_index_search_tag_for_var(ident)
                c_name = util.parsing.mangle_fortran_var_expr(var_expr) 
                substitute = ttnode.fstr().replace(ident,c_name)
                ttnode.overwrite_cstr(substitute)
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
                # todo: 
                # search through the subtree and ensure that only
                # the last element is an array indexed by the loop
                # variables 
                # Deep copy required for expressions that do not match
                # this criterion
                if (":" in ident 
                   or "(" in ident): # todo: hack
                    raise util.error.LimitationError("cannot map expression '{}'".format(ident))
                var_expr = indexer.scope.create_index_search_tag_for_var(ident)
                c_name = util.parsing.mangle_fortran_var_expr(var_expr) 
                substitute = ttnode.fstr().replace(ident,c_name)
                ttnode.overwrite_cstr(substitute)
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
        if isinstance(value._value, tree.TTTensorEval):
            if ident.startswith("_"):
                # function introduced by GPUFORT, Fortran identifiers never start with '_'
                value._value._type = tree.TTTensorEval.Type.FUNCTION_CALL
            else:
                value_type, _ = indexer.scope.search_scope_for_value_expr(scope, ident)
                if value_type == indexer.indexertypes.ValueType.VARIABLE:
                    value._value._type = tree.TTTensorEval.Type.ARRAY_ACCESS
                elif value_type == indexer.indexertypes.ValueType.PROCEDURE:
                    value._value._type = tree.TTTensorEval.Type.FUNCTION_CALL
                elif value_type == indexer.indexertypes.ValueType.INTRINSIC:
                    value._value._type = tree.TTTensorEval.Type.INTRINSIC_CALL