# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
# configurable parameters
log_prefix = "indexer"

structures = r"module|program|function|routine|procedure|subroutine|interface|type|(end\s*(module|program|function|subroutine|interface|type))"
declarations = r"integer|real|double|logical" # derived types already considered by STRUCTURES
attributes = r"attributes"
use = r"use"
directives = r"([!c\*]\$\w+)"

filter = r"\b(" + structures + "|" + declarations + "|" + attributes + "|" + use + r")\b" + "|" + directives

continuation_filter = r"(\&\s*\n)|(\n\s*[\!c\*]\$\w+\&)"

pretty_print_index_file = False # Pretty print index before writing it to disk.

parse_var_declarations_worker_pool_size = 1 # Number of worker threads for parsing variable declarations.
parse_var_modification_statements_worker_pool_size = 1 # Number of worker threads for parsing statements that modify variable index linemaps, e.g. CUDA Fortran attributes statements  or OpenACC acc declare directives.

module_ignore_list = [ # these modules are ignored when checking dependencies
    "cudafor", "cublas", "cusparse", "cusolver", "iso_c_binding",
    "iso_fortran_env"
]

scopes = [] # Can be used to preload scopes
remove_outdated_scopes = False # Set to false if you want to preload scopes via your config file
