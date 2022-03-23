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
    "iso_c_binding",
    "iso_fortran_env", 
    "cudafor",
    "cublas",
    "cusparse",
    "cusolver", 
    "hipfort", 
    "hipfort_check",
    "hipfort_hipblas",
    "hipfort",
    "hipfort_auxiliary",
    "hipfort_check",
    "hipfort_cuda_errors",
    "hipfort_enums",
    "hipfort_hipblas",
    "hipfort_hipblas_auxiliary",
    "hipfort_hipblas_enums",
    "hipfort_hipfft",
    "hipfort_hipfft_enums",
    "hipfort_hipfft_params",
    "hipfort_hiphostregister",
    "hipfort_hipmalloc",
    "hipfort_hipmemcpy",
    "hipfort_hiprand",
    "hipfort_hiprand_enums",
    "hipfort_hipsolver",
    "hipfort_hipsolver_enums",
    "hipfort_hipsparse",
    "hipfort_hipsparse_enums",
    "hipfort_rocblas",
    "hipfort_rocblas_auxiliary",
    "hipfort_rocblas_enums",
    "hipfort_rocfft",
    "hipfort_rocfft_enums",
    "hipfort_rocrand",
    "hipfort_rocrand_enums",
    "hipfort_rocsolver",
    "hipfort_rocsolver_enums",
    "hipfort_rocsparse",
    "hipfort_rocsparse_enums",
    "hipfort_types",
]

gpufort_module_file_suffix = ".gpufort_mod"

scopes = [] # Can be used to preload scopes
remove_outdated_scopes = False # Set to false if you want to preload scopes via your config file
