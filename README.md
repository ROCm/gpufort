# 1. Porting Fortran OpenACC applications to Fortran + HIP with GPUFORT

<!-- TOC -->

- [1. Porting Fortran OpenACC applications to Fortran + HIP with GPUFORT](#1-porting-fortran-openacc-applications-to-fortran--hip-with-gpufort)
  - [1.1. Scenario](#11-scenario)
  - [1.2. Installing GPUFORT](#12-installing-gpufort)
    - [1.2.1. Install latest HIPFORT](#121-install-latest-hipfort)
    - [1.2.2. Install extended OpenACC Fortran interfaces](#122-install-extended-openacc-fortran-interfaces)
    - [1.2.3. Installing GPUFORT](#123-installing-gpufort)
      - [1.2.3.1. GPUFORT components and dependencies](#1231-gpufort-components-and-dependencies)
  - [1.3. Using GPUFORT](#13-using-gpufort)
    - [1.3.1. Basic Usage](#131-basic-usage)
  - [1.4. Setting up Dynamico](#14-setting-up-dynamico)
  - [1.5. Using GPUFORT](#15-using-gpufort)
    - [1.5.1. Discover files that contain OpenACC code](#151-discover-files-that-contain-openacc-code)
    - [1.5.2. Run](#152-run)

<!-- /TOC -->

## 1.1. Scenario

We have an OpenACC application that can be built with 
NVIDIA's HPC Fortran compiler.
In this guide, our goal is to port the
open-source HPC application Dynamico to 
Fortran + HIP (C++), where the host-side
of the code should be written in Fortran and the device-side
computations should be written in HIP C++.

## 1.2. Installing GPUFORT

> **Required time:** 5-10 minutes

GPUFORT itself is a python3 tool with a bash frontend. It depends on a number of third-party python packages
that can usually be installed via the python package installer `pip` 
on mo st operating systems. We will revisit this later.
GPUFORT *applications* that are ported to Fortran + HIP have a number of other dependencies
that must be installed on the system running the applications.
Before we discuss how to install GPUFORT itself, we
will show how the dependencies are installed. 

### 1.2.1. Install latest HIPFORT

The first dependency we install is HIPFORT.
HIPFORT provides Fortran interfaces to the HIP runtime
as well as to the HIP and ROCm math libraries.
This is if an application is compiled for AMD GPUs.
If the application is compiled for NVIDIA GPUs, HIPFORT
delegates to the CUDA runtime and CUDA math libraries. 

> **NOTE:** The HIPFORT master branch is usually a little further ahead than
the packages from the ROCm package repositories. Moreover, the format of the Fortran module files shipped with the latter packages might not be 
compatible with your preferred Fortran compiler.
Currently, we thus recommended to download and build HIPFORT yourself.

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

### 1.2.2. Install extended OpenACC Fortran interfaces

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

> **NOTE:** As long as the repository remains private, AMD single-sign-on users need to clone the repository via:

```bash
GIT_SSH_COMMAND='ssh -i <private_key_file> -o IdenitiesOnly=yes' git clone git@github.com:RocmSoftwarePlatform/openacc-fortran-interfaces.git
```

The cloned repository has the following structure:

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

The folder `openacc_gomp` contains the sources and Makefile to build extended
interfaces to GCC's LIBGOMP. 
The folder `gpufort` contains a self-written runtime.
Both can be build by changing into the subfolder and calling `make`.
(Do not specify `-j` option!)
The `rules.mk` can be adapted to switch on debug compiler flags
and additional debugging features of the respective runtime.

### 1.2.3. Installing GPUFORT

Finally, we install GPUFORT itself. To this end, we clone the repository to our preferred installation folder:

```bash
git clone git@github.com:RocmSoftwarePlatform/gpufort.git
```

> **NOTE:** As long as the repository remains private, AMD single-sign-on users need to clone the repository via:

```bash
GIT_SSH_COMMAND='ssh -i <private_key_file> -o IdenitiesOnly=yes' git clone git@github.com:RocmSoftwarePlatform/gpufort.git
```

Installation is completed by pointing the `PATH` environment variable
to GPUFORT's `bin/` subfolder. For example via:
```
export PATH=$PATH:<gpufort_install_dir>/bin
```
If you don't want to run this command everytime, add it to a startup script
such as `/home/<user>/.bashrc`.


#### 1.2.3.1. GPUFORT components and dependencies

The essential parts of the GPUFORT directory are displayed below:

```
gpufort
├── bin
│   ├── gpufort
│   └── gpufort-indexer
├── config
│   └── dynamico.py.in
├── examples
...
├── os-requirements.txt
├── python
...
├── python3-requirements.txt
├── README.md
...
```

The `bin/` subfolder contains the two main GPUFORT tools:

| Name                  | Description |
|-----------------------|-------------|
| `gpufort`             | The main GPUFORT tool, the source-to-source translator. |
| `gpufort-indexer`     | A separate indexer tool for discovering accelerator code/directives in Fortran files. Its output can be used by `gpufort`. |

All system and `python3` requirements are listed in the files
`os-requirements.txt` and `python3-requirements.txt`, respectively.
In particular, GPUFORT users must have  `bash`, `python3` (>= 3.6), `gfortran`, and `clang-format` (> version 6.0) installed on their machine. They further need to install the `python3` packages: `jinja2` , `pyparsing`,
and `fprettify`.

> **NOTE:** The tools in the `bin/` folder check if all GPUFORT dependencies are installed and will tell you which ones are missing as error message. They also suggest installation commands for missing `python3` modules.

* The `examples/` subfolder contains OpenACC (`openacc`) and CUDA Fortran (`cuf`)
examples, which demonstrate how to use the GPUFORT tools.
* The `python` subfolder contains: 
  * The `pyparsing` Fortran 2003, CUDA Fortran, and OpenACC grammar in the `grammar` subdirectory. 
  * `translator` contains the `python3` sources for the
translatior (abstract syntax) tree (*TT*). This tree is used
to analyze Fortran constructs and directives. Furthmore,
it is used to generate (HIP) C++ code from the parsed Fortran
code blocks.
  * The `scanner` subfolder contains a separate less sophisticated 
parser for detecting code lines in the translation source
that are considered interesting for the translation process
because they must be translated or they contain information
that is interesting for the translation of other code lines.
  * The `indexer` subfolder contains a slimmed down version of the `scanner`
for discovering interesting declarations and directives in 
Fortran source directories provided by the user. The discovery relies internally on `gfortran`. Arguments such as `-Dsymbol=definition` are forwarded to this backend, i.e specific compilation paths can be understood.

All three `scanner`, `translator` and `indexer` are built on top of the
`pyparsing` `grammar`. Modules `indexer` and `scanner` rely on module `translator` for in-depth analysis of discovered code lines.

## 1.3. Using GPUFORT

### 1.3.1. Basic Usage

You can call `gpufort` as follows if you agree with the default
configuration:

```
gpufort <fortran_file>
```

Typically this will not be the case. 
Fortunately, `gpufort` provides multiple ways to change the default behaviour:

* You can supply a `python3` configuration file,
* you can use a number command line arguments,
* you can register a source-2-source translation backend,
* you can edit the `jinja2` code generation templates
  when generating HIP C++ code from, or
* you can edit the python sources.

The default configuration parameters can be printed out as follows:

```
gpufort --print-config-defaults
```

Sample output (shortened):

```

```

## 1.4. Setting up Dynamico

> TBA

> SHOW TREE


## 1.5. Using GPUFORT

### 1.5.1. Discover files that contain OpenACC code

This can be done via grep, e.g. via:

### 1.5.2. Run 
