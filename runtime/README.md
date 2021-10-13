# Fortran interfaces to OpenACC runtimes

* This directory provides extensions to the `openacc.f90` Fortran
  interfaces for OpenACC runtimes.
  In particular, it provides a bridge to runtime functions that
  realise `!$acc enter data`, `!$acc exit data`, and `!$acc data`.
  Moreover, a bridge to `acc_deviceptr` is provided, which
  is often not found in `openacc.f90`.
 
  Features:
  * Thread-safe
  * Standard-compliant runtime API
 
  Limitations:
  * This runtime does not support the `acc_get_cuda_stream` vendor-specific extension.
    Therefore, no `hipStream` instance can be obtained from the runtime and passed to 
    `hipfort` API calls or `GPUFORT` kernel launch routines.
  * Examples are likely outdated 

> **NOTE:** Currently only interfaces to GCC's LIBGOMP are provided.

* This directory further contains `gpufort_acc_runtime`, a minimal non-standard-compliant runtime written completely
  in Fortran, which can be used for teaching/training or prototyping.
  
  Features:
  * `hipStream` instances can be obtained from the runtime.
  Limitations:
  * Not thread-safe
  * No standard-compliant runtime API (might change in the future) 

## Runtime subfolders

* `openacc_gomp`: contains extended interfaces to GCC's LIBGOMP.
* `gpufort_acc_runtime`: contains a minimal non-standard-compliant runtime written 
  completely in Fortran, which can be used teaching purposes
* `openacc_hpe` ???

## Building

```
cd <runtime_subfolder>
make
```

Take a look into the `Makefile` in each folder and adapt it to your needs.
Certain Makefile variables can be overwritten via the environment variables  
`FC`, `HIPFORT_PATH`, `FCFLAGS`, `CXX`, `CXXFLAGS`.

Example:

```
cd <runtime_subfolder>
FC=/usr/bin/gfortran HIPFORT_PATH=/opt/rocm/hipfort/install/ make clean all
```

> **NOTE:** `python3` with `jinja2` package installed is required to use the codegen target.`

## Outlook

* We require a standard-compliant, thread-safe HIP C++ based runtime that can
  be used/built for HIP devices from different vendors (AMD, NVIDIA) to
  support efforts such as GPUFORT. Main purpose of the runtime must be
  mapping data between host and device and managing HIP streams.
  Other features are optional.
