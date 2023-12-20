# Fortran interfaces to OpenACC runtimes

* This directory provides extensions to the `openacc.f90` Fortran
  interfaces for OpenACC runtimes.
  In particular, it provides a bridge to runtime functions that
  realise `!$acc enter data`, `!$acc exit data`, and `!$acc data`.
  Moreover, a bridge to `acc_deviceptr` is provided, which
  is often not found in `openacc.f90`.
 
  * Currently only interfaces to GCC's LIBGOMP are provided.

    Features:
    * Thread-safe
    * Standard-compliant runtime API

    Limitations:
    * This runtime does not support the `acc_get_cuda_stream` vendor-specific extension.
      Therefore, no `hipStream` instance can be obtained from the runtime and passed to 
      `hipfort` API calls or `GPUFORT` kernel launch routines.
    * Examples are likely outdated 

* This directory further contains `gpufortrt`, a minimal non-standard-compliant runtime written 
  mostly in Fortran, which we use for prototyping and characterizing application runtime behavior. 
  Long term, we plan to rewrite this runtime in C++ or abandon it for a better alternative.
 
  Features:
  * `hipStream` instances can be obtained from the runtime.
  Limitations:
  * No thread-safe access (unless all required arrays are present)
  * No standard-compliant runtime API (might change long term) 

  Requirements:
  * `python3` with `jinja2` package is required to generate some of the interfaces
     this runtime is using.
  * Non-standard GNU Fortran extension `sizeof` must be supported by the used Fortran compiler
  * Assumed-type, assumed-rank arrays must be supported by the used Fortran compiler


## Runtime subfolders

* `openacc_gomp`: contains extended interfaces to GCC's LIBGOMP.
* `gpufortrt`: contains a minimal non-standard-compliant runtime.

## Building

```
cd <runtime_subfolder>
make
```

Take a look into the `Makefile` in each folder and adapt it to your needs.
Certain Makefile variables can be overwritten via environment variables.


