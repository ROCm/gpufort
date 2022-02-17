# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
# Defaults
def GET_DEFAULT_BLOCK_DIMS(kernel_name, dim):
    block_dims = {1: [128], 2: [128, 1, 1], 3: [128, 1, 1]}
    return block_dims[dim]


def GET_DEFAULT_LAUNCH_BOUNDS(kernel_name):
    return None


# OPTIONS
log_prefix = "fort2x.hip.codegen"
# Prefix for log output that this component writes.
get_block_dims = GET_DEFAULT_BLOCK_DIMS
# Callback to provide default block dimensions for a given kernel.
# callback arguments: kernel_name,file_path,lineno,dim
# return: list of int with dim entries
get_launch_bounds = GET_DEFAULT_LAUNCH_BOUNDS
# Callback to provide 'MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP' for a given kernel.
# callback arguments: kernel_name,file_path,lineno
# return: a string consisting of two comma-separated integer numbers, e.g. '128,1' or '256, 4'
emit_gpu_kernel = True
# Emit GPU kernels, defaults to True.
emit_grid_launcher = True
# Only for loop nests: Render a launcher that takes the grid as first argument, defaults to True.
# This launcher is always rendered for kernel procedures, i.e. this option
# does not affect them.
emit_problem_size_launcher = True
# Only for loop nests: Render a launcher that takes the problem size as first argument, defaults to True.
emit_fortran_interfaces = True
# Emit explict Fortran interfaces to the kernel launchers, defaults to True.
emit_cpu_launcher = False
# Generate CPU kernel launch routines from Fortran loop nests. (EMIT_KERNEL_LAUNCHER must be set to True too.)
emit_debug_code = False
# Generate debug routine calls into the code that can be used
# to print out kernel argument values or device array elements and norms.
