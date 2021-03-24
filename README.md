# Porting Fortran OpenACC applications to Fortran + HIP with GPUFORT

## Scenario

We have an OpenACC application that can be built with 
NVIDIA's HPC Fortran compiler.
In this guide, our goal is to port the
open-source HPC application Dynamico to 
Fortran + HIP (C++), where the host-side
of the code should be written in Fortran and the device-side
computations should be written in HIP C++.

## Installing GPUFORT

GPUFORT itself is a python3 tool with a bash frontend. It depends on a number of third-party python packages
that can usually be installed via the python package installer `pip` 
on most operating systems. We will revisit this later.
GPUFORT *applications* that are ported to Fortran + HIP have a number of other dependencies
that must be installed on the system running the applications.
Before we discuss how to install GPUFORT itself, we
will show how the dependencies are installed. 

### Install latest HIPFORT

The first dependency we install is HIPFORT.
HIPFORT provides Fortran interfaces to the HIP runtime
as well as to the HIP and ROCm math libraries.
This is if an application is compiled for AMD GPUs.
If the application is compiled for NVIDIA GPUs, HIPFORT
delegates to the CUDA runtime and CUDA math libraries. 

To install HIPFORT, we first download the sources via:

```bash
git clone git@github.com:RocmSoftwarePlatform/hipfort.git
```

Next we install HIPFORT to `/opt/rocm/hipfort` (recommended) via `cmake`.
As usually with `cmake`, we first create a `build` subfolder in the hipfort directory:

```bash
cd hipfort
mkdir build
```

The HIPFORT installation will then have the following directory structure:

```
hipfort/
├── bin
├── build
├── cmake
│   └── Modules
├── lib
│   ├── hipfort
│   └── modules-amdgcn
└── test
    ├── f2003
    └── f2008
```

We enter the `build` subfolder and install HIPFORT via:

```bash
cd build
cmake -DHIPFORT_INSTALL_DIR=/opt/rocm/hipfort ../
make -j<num_proc>
make install
```

Finally, we add the HIPFORT **installation's** `bin` directory to the `PATH`:

```bash
export PATH=$PATH:/opt/rocm/hipfort/bin
```

We can test the `PATH` setting via:

```bash
hipfc -h
```

We can test HIPFORT itself by following the steps in

https://github.com/ROCmSoftwarePlatform/hipfort/blob/master/README.md

## Install extended OpenACC Fortran interfaces

The OpenACC programming model requires that any CPU thread can map host arrays
to the device to run certain computations on the device and vice versa. 
Such a mapping might be, e.g., an allocation of device array with the
same shape of the host array or a copy of the host array. This list
is not complete.
Therefore, GPUFORT generally needs a runtime to port OpenACC applications
to HIP C++. One example runtime that can be used for this purpose is 
GCC's LIBGOMP.
Unfortunately, available OpenACC runtimes, such as LIBGOMP, do often not expose all needed functionality
to GPUFORT.
While the OpenACC standard defines runtime routines that can be mapped directly to
many OpenACC directives and constructs, it does not so for data constructs and directives such
as such as `acc data`, `acc enter data`, and `acc exit data`.
In addition, there are functions such as `acc_deviceptr` that are only available to C programmers
but not to Fortran programmers.
In order to make GPUFORT and other Fortran programmers or tools still rely on this 'hidden' runtime functionality, 
we built interfaces to internal/non-exposed routines of OpenACC runtimes such as LIBGOMP. 
We further implemented a simple runtime ourself, which we will use in this guide as runtime for the ported
HPC application.
The interfaces and the simple runtime can be obtained via:

```bash
git clone git@github.com:RocmSoftwarePlatform/openacc-fortran-interfaces.git
```

> NOTE: As long as the repository remains private, AMD single-sign-on users need to clone the repository via:

```bash
GIT_SSH_COMMAND='ssh -i <private_key_file> -o IdenitiesOnly=yes' git clone git@github.com:RocmSoftwarePlatform/openacc-fortran-interfaces.git
```

After downloading the 

```
openacc-fortran-interfaces/
├── examples
│   ├── gpufort_acc_runtime
│   └── openacc_gomp
├── gpufort_acc_runtime
│   ├── codegen
│   ├── Makefile
│   ├── rules.mk
│   ├── src
│   ├── src_mt
│   └── test
├── LICENSE
├── openacc_gomp
│   ├── codegen
│   ├── Makefile
│   ├── rules.mk
│   ├── src
│   └── studies
└── README.md
```

#### Download from private repositories

As long as both repositories are private, we need to use the following 
git clone commands:

```bash
GIT_SSH_COMMAND='ssh -i <private_key_file> -o IdenitiesOnly=yes' git clone git@github.com:RocmSoftwarePlatform/gpufort.git
```

* Download GPUFORT

```bash
git clone git@github.com:RocmSoftwarePlatform/gpufort.git
```

```
gpufort
├── bin
│   ├── gpufort
│   └── gpufort-indexer
├── config
│   └── dynamico.py.in
├── examples
...
├── LICENSE
├── os-requirements.txt
├── python
...
├── python3-requirements.txt
├── README.md
...
```


#### Installing GPUFORT dependencies



#### Building OpenACC Fortran interfaces

It is essential that we 

## Setting up Dynamico

> TBA

> SHOW TREE


## Using GPUFORT

### 1. Discover files that contain OpenACC code

This can be done via grep, e.g. via:

### 2. Run 
