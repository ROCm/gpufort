# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
log_prefix = "scanner" # log prefix

translation_enabled_by_default = True

source_dialects     = set(["cuf", "acc"]) # one of ["acc","cuf","omp"]
destination_dialect = "omp" # one of ["omp","hip-runtime-rt"]

kernels_to_convert_to_hip = [
] # add line number here (might change if snippets are included included); check log to find actual line number

loop_vars = "integer :: {}".format(",".join([
    "_" + chr(ord("a") + i) for i in range(0, 20)
])) # integer :: _a,_b,_c,_d,...

loop_kernel_name_template = "{parent}_{lineno}" # parent: name of parent module,program,subroutine (lower case)
# lineno: line number
# hash: Hash of the kernel (whitespaces are removed beforehand)

loop_kernel_default_launcher = "auto" # "auto" or "cpu"

hip_module_name = "hipfort"
hip_math_module_prefix = hip_module_name + "_"

cuda_ifdef = "CUDA"
cublas_version = 1
keep_cuda_lib_names = False

line_groups_enable = True # group modified lines such that they appear in the block when wrapping them in ifdefs.
line_groups_include_blank_lines = True # Include intermediate blank lines into a line group.
