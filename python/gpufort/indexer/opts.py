
        # configurable parameters
log_prefix = "indexer"

modern_fortran = True
        # Expect modern Fortran comments and directives
cuda_fortran = True
        # Expect CUDA Fortran statements, directives and conditional code such as `!@cuf ierr = cudaDeviceSynchronize()`
openacc = True
        # Expect OpenACC directives and runtime calls 

default_module_accessibility = "public"
        # if no private/public statement is found
        # in module, this default value is assumed.

default_type_accessibility = "public"
        # if no private/public statement is found
        # in module, this default value is assumed.

module_ignore_list = [ # these modules are ignored when checking dependencies
    "iso_c_binding",
    "iso_fortran_env", 
    "ieee_arithmetic",
    "openacc",
    "openacc_lib",
    "mpi",
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
    "netcdf",
]

pretty_print_index_file = False # Pretty print index before writing it to disk.

gpufort_module_file_suffix = ".gpufort_mod"

scopes = [] # Can be used to preload scopes
