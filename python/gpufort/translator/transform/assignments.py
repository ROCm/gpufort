# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
"""
Collection of routines for generating loopnests (nested tree.TTDo instances) out of
Fortran array expressions.

Concepts:

* Uses the `is_contiguous` property of array expressions to 
  map an array expression to a single loop where
  all elements are acessed via the [] operator.
  * If the expression is a subarray and not a full array,
    an offset needs to be added to the index.
    * To not evaluate the offset during every iteration,
      a constant variable reference is used within the loop.
  * If an array is not contiguous, it will be mapped to a loopnest.

:note: Uses the `loops` package's `unique_label` routine
       to create unique identifiers.
       This routine relies on a global counter that
       can be reset via the the package's `reset` method.
:note: Uses the `loops` package's `render_const_int_decl`
       routine to render int constant declarations.
"""

from .. import tree

from . import loops
  
def _render_c_idx_offset(cname,indices):
    """Computes index offset for a contiguous array.
    """
    return "{cname}.c_idx({args})".format(
      cname = cname,
      args = ",".join(indices)
    )

def generate_loopnest_for_contiguous_array(ttvalue,yields_full_array):
    """Generate a loopnest based on the 
    :return: A list with a single entry: 
             A dummy TTDo that can be used to generate C++ code
             and C++/Fortran problem size information.
    :note: The procedure assumes that the Fortran representation of the do loop
           will only be used to calcuate the index range in Fortran code. 
    :note: Contiguous arrays may not contain index ranges with stride > 1.
    :note: Assumes that the index ranges come first,
           as guaranteed by the semantics check.
    :note: Assumes that only the first index range may have
           a specified lbound and only 
           the last index range may have an upper bound.
    :note: Assumes that the array is represented as C++ type
           with a function `c_idx(int i1, ..., int i_rank)` 
           and `lbound(dim)`. 
    :note: Uses the cname member of the TTValue object as the expressions original
           name may have been overriden with a particular name.
           This would, e.g., be the case if members of a derived type are passed separately to a kernel
           instead of the full derived type.
    """
    cname = ttvalue.cname
    if yields_full_array:
        last_cstr = "{}.size() - 1".format(cname)
    else:
        (lbounds, ubounds, steps) = ttvalue.loop_bounds_as_str(
          only_consider_index_ranges=False,
          converter = tree.traversals.make_cstr
        ) 
        first_cidx = _render_c_idx_offset(cname,lbounds)
        last_cidx  = _render_c_idx_offset(cname,ubounds)
        last_cstr = last_cidx + " - " + first_cidx + " + 1"
    # create dummy do loop
    first = tree.TTDummy(
      fstr = "1",
      cstr = "0" 
    )
    last = tree.TTDummy(
      fstr = "size({})".format(ttvalue.fstr()),
      cstr = last_cstr 
    )
    index = tree.TTIdentifier([loops.unique_label("idx")])
    begin = tree.TTAssignment([index,first])
    step = None
    body = []
    return [
      tree.TTDo([begin,last,step,body])
    ]

def generate_loopnest_for_generic_subarray(ttvalue):
    """Loopnest generator generic subarray expressions.
    :return: A nest of dummy TTDos that can be used to generate C++ code
             Fortran problem size information. 
    """
    loop_list = []
    (lbounds_as_cstr, 
    ubounds_as_cstr, 
    steps_as_cstr) = ttvalue.loop_bounds_as_str(
      True,tree.traversals.make_cstr
    ) 

    (lbounds_as_fstr, 
    ubounds_as_fstr, 
    steps_as_fstr) = ttvalue.loop_bounds_as_str(
      True,tree.traversals.make_fstr
    ) 

    # create dummy do-loop nest, go from inner to outer loop, left to right
    for (lbound_as_fstr,
         ubound_as_fstr,
         step_as_fstr,
         lbound_as_cstr,
         ubound_as_cstr,
         step_as_cstr) in zip(
           lbounds_as_fstr,
           ubounds_as_fstr,
           steps_as_fstr,
           lbounds_as_cstr,
           ubounds_as_cstr,
           steps_as_cstr
        ):
        first = tree.TTDummy(
          cstr = lbound_as_cstr,
          fstr = lbound_as_fstr
        )
        last = tree.TTDummy(
          cstr = ubound_as_cstr,
          fstr = ubound_as_fstr
        )
        index = tree.TTIdentifier([loops.unique_label("idx")])
        begin = tree.TTAssignment([index,first])
        if step_as_fstr != None:
            step = tree.TTDummy(
              cstr = ubound_as_cstr,
              fstr = ubound_as_fstr
            )
        else:
            step = None
        body = []
        loop_list.append(tree.TTDo([begin,last,step,body]))
    return loop_list

def nest_loop_list(loops,innermost_loop_body):
    """Nests a linear list of loops so that the do-loop traversal algorithm can be used.
    :param list loops: A list of loops.
    :param list innermost_loop_body: A list of statement nodes to put into
                                       the body of the innermost loop.
    :return: Reference to the first loop.
    """
    assert len(loops)
    first = loops[0]
    last = loops[0]
    for loop in loops[1:]:
        last.body.append(loop)
        last = loop
    last.body += innermost_loop_body
    return first  

def modify_c_repr_of_contiguous_array_expr(ttassignment,loop_index_cstr):
    """Overwrite TTValue nodes that have originally a Fortran-style () index operator to use
    a C-style [] index operator for accessing the array elements.
    For contiguous subarrays that do not cover the full array, an
    offset is stored.
    :return: A list of const int declarations per offsets
             of discovered contiguous subarrays.
    """
    # TODO
    #assert (ttassignment.rhs.yields_contiguous_array
    #       or ttassignment.rhs.yields_scalar)
    #assert not ttassignment.rhs.applies_transformational_functions_to_arrays
    #assert not ttassignment.rhs.contains_array_access_with_index_array
    offset_var_decls = [] 
    for ttnode in ttassignment.walk_preorder():
        if isinstance(ttnode,tree.TTValue):
            if ttnode.is_contiguous_array:
                if ttnode.is_full_array:
                    array_index_cstr = loop_index_cstr
                else: 
                    (lbounds,ubounds,steps) = ttnode.loop_bounds_as_str(
                      False,tree.traversals.make_cstr
                    ) 
                    offset_var = loops.unique_label("offset") 
                    offset_cstr = _render_c_idx_offset(ttnode.cname,lbounds)
                    offset_var_decls.append(loops.render_const_int_decl(
                      offset_var,offset_cstr)
                    )
                    array_index_cstr = loop_index_cstr + " + " + offset_var
                ttnode.overwrite_c_repr(
                  ttnode.cname,
                  [array_index_cstr],
                  True # square brackets
                )
    return offset_var_decls

def modify_c_repr_of_generic_array_expr(ttassignment,loop_indices):
    """Overwrite TTValue nodes that
    """
    # TODO
    #assert (ttassignment.rhs.yields_contiguous_array
    #       or ttassignment.rhs.yields_scalar)
    #assert not ttassignment.rhs.applies_transformational_functions_to_arrays
    #assert not ttassignment.rhs.contains_array_access_with_index_array
    for ttnode in ttassignment.walk_preorder():
        if isinstance(ttnode,tree.TTValue):
            rank_defining_node = ttnode.rank_defining_node
            if rank_defining_node.is_array:
                loop_indices_copy = list(loop_indices)
                cargs = []
                for arg in rank_defining_node.args:
                    (lbounds,ubounds,steps) = ttnode.loop_bounds_as_str(
                      True,tree.traversals.make_cstr
                    )
                    if isinstance(arg,tree.TTIndexRange):
                        cargs.append(loop_indices_copy.pop(0))
                    else:
                        cargs.append(arg.cstr())
                ttnode.overwrite_c_repr(
                  ttnode.cname,
                  cargs,
                  False # no square brackets
                )
