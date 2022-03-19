# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import util
from . import base

def move_statements_into_loopnest_body(ttloopnest):
    # subsequent loop ranges must not depend on LHS of assignment
    # or inout, out arguments of function call
    pass 

def collapse_loopnest(do_loops):
    indices = []
    problem_sizes = []
    for loop in do_loops:
        problem_sizes.append("({})".format(
            loop.problem_size(converter=base.make_c_str)))
        loop.thread_index = "_remainder,_denominator" # side effects
        indices.append(loop.collapsed_loop_index_c_str())
    conditions = [ do_loops[0].hip_thread_bound_c_str() ]
    indices.insert(0,"int _denominator = {};\n".format("*".join(problem_sizes)))
    indices.insert(0,"int _remainder = __gidx1;\n")
    return indices, conditions

def map_loopnest_to_grid(do_loops,num_outer_loops_to_map):
    if num_outer_loops_to_map > 3:
        util.logging.log_warn(
            "loop collapse strategy 'grid' chosen with nested loops > 3")
    num_outer_loops_to_map = min(3, num_outer_loops_to_map)
    thread_indices = ["x", "y", "z"]
    for i in range(0, 3 - num_outer_loops_to_map):
        thread_indices.pop()
    indices = []
    conditions = []
    for loop in do_loops:
        if not len(thread_indices):
            break
        loop.thread_index = thread_indices.pop()
        indices.append(loop.hip_thread_index_c_str())
        conditions.append(loop.hip_thread_bound_c_str())
    return indices, conditions
