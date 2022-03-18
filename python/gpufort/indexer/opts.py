        # SPDX-License-Identifier: MIT
        # Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
        # configurable parameters
log_prefix = "indexer"

modern_fortran = True
        # Expect modern Fortran comments and directives
cuda_fortran = True
        # Expect CUDA Fortran statements, directives and conditional code such as `!@cuf ierr = cudaDeviceSynchronize()`
openacc = True
        # Expect OpenACC directives and runtime calls 

pretty_print_index_file = False # Pretty print index before writing it to disk.

module_ignore_list = [ # these modules are ignored when checking dependencies
    "cudafor", "cublas", "cusparse", "cusolver", "iso_c_binding",
    "iso_fortran_env"
]

gpufort_module_file_suffix = ".gpufort_mod"

scopes = [] # Can be used to preload scopes
remove_outdated_scopes = False # Set to false if you want to preload scopes via your config file
