#  GPUFORT installation and basic usage

- [GPUFORT installation and basic usage](#gpufort-installation-and-basic-usage)
  - [1. GPUFORT installation](#1-gpufort-installation)
    - [1.1. Install latest HIPFORT](#11-install-latest-hipfort)
    - [1.2. Install extended OpenACC Fortran interfaces](#12-install-extended-openacc-fortran-interfaces)
    - [1.3. Install GPUFORT tools](#13-install-gpufort-tools)
      - [1.3.1. GPUFORT components and dependencies](#131-gpufort-components-and-dependencies)
  - [2. Using and configuring GPUFORT](#2-using-and-configuring-gpufort)
    - [2.1. Command line options](#21-command-line-options)
    - [2.2. Configuration files](#22-configuration-files)
    - [2.3. Change source and destination dialects](#23-change-source-and-destination-dialects)
    - [2.4. Only modify host file or only generate kernels](#24-only-modify-host-file-or-only-generate-kernels)
    - [2.5. Control S2S translation via GPUFORT directives](#25-control-s2s-translation-via-gpufort-directives)
  - [3. Debugging and optimizing GPUFORT HIP C++ kernels](#3-debugging-and-optimizing-gpufort-hip-c-kernels)
    - [3.1. Synchronize all or only particular kernels](#31-synchronize-all-or-only-particular-kernels)
    - [3.2. Print input argument values](#32-print-input-argument-values)
    - [3.3. Print device array elements and metrics](#33-print-device-array-elements-and-metrics)
    - [3.4. An kernel optimizing workflow](#34-an-kernel-optimizing-workflow)

## 1. GPUFORT installation

> **Required time:** 5-10 minutes

GPUFORT itself is a `python3` tool (collection) with a bash frontend. It depends on a number of third-party `python3` packages that can usually be installed via the `python` / `python3` package installer `pip`. 
*Applications* that are ported to Fortran + HIP via GPUFORT, in contrast, have a number of additional dependencies that must be installed on the system running those applications.
We will discuss how the dependencies are installed. 

### 1.1. Install latest HIPFORT

The first dependency that we install is HIPFORT.
HIPFORT provides Fortran interfaces to the HIP runtime
and the HIP and ROCm math libraries.
(If the application is compiled to run on NVIDIA GPUs, HIPFORT
delegates to the CUDA runtime and math libraries.)

> **NOTE:** The HIPFORT master branch is often a little further ahead than
the packages from the ROCm package repositories. Moreover, the format of the Fortran module files shipped with the latter packages might not be 
compatible with your preferred Fortran compiler.
Currently, we thus recommended to download and build HIPFORT yourself.

To install HIPFORT, we first download the sources:

```bash
git clone git@github.com:RocmSoftwarePlatform/hipfort.git
```

Next we install HIPFORT to `/opt/rocm/hipfort` (recommended) via `cmake`.
As usually with `cmake`, we first create a `build` subfolder in the hipfort directory:

```bash
cd hipfort
mkdir build
```

The HIPFORT installation will have the structure:

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

We enter the `build` subfolder and install HIPFORT:

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

We can test HIPFORT itself by following the steps in the HIPFORT 
`README.md` file:

https://github.com/ROCmSoftwarePlatform/hipfort/blob/master/README.md

### 1.2. Install extended OpenACC Fortran interfaces

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
We further implemented a simple runtime ourselves, which we will use in this guide as runtime for the ported
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

### 1.3. Install GPUFORT tools

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
If you don't want to run this command every time, add it to a startup script
such as `/home/<user>/.bashrc`.


#### 1.3.1. GPUFORT components and dependencies

The essential parts of the GPUFORT directory are shown below:

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
translator (abstract syntax) tree (*TT*). This tree is used
to analyze Fortran constructs and directives. Furthermore,
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

## 2. Using and configuring GPUFORT

You can call `gpufort` as follows if you agree with the default
configuration:

```
gpufort <fortran_file>
```

Typically this will not be the case. 
Fortunately, `gpufort` provides multiple ways to change the default behavior:

1. You can use a number of command line arguments.
2. You can supply a `python3` configuration file.
3. You can register a source-2-source translation backend.
4. You can edit the `jinja2` code generation templates.
   when generating HIP C++ code.
5. You can edit the python sources.

### 2.1. Command line options

You can list command line options via `-h` or `--help`:

```
gpufort -h
```

Output:
```
usage: gpufort.py [-h] [-o,--output O,__OUTPUT] [-d,--search-dirs [SEARCHDIRS [SEARCHDIRS ...]]] [-i,--index INDEX] [-w,--wrap-in-ifdef] [-k,--only-generate-kernels]
                  [-m,--only-modify-host-code] [-E,--destination-dialect DESTINATIONDIALECT] [--log-level LOGLEVEL] [--cublas-v2] [--working-dir WORKINGDIR] [--print-config-defaults]
                  [--config-file CONFIGFILE]
                  [input]

S2S translation tool for CUDA Fortran and Fortran+X

positional arguments:
  input                 The input file.

optional arguments:
  -h, --help            show this help message and exit
  -o,--output O,__OUTPUT
                        The output file. Interface module and HIP C++ implementation are named accordingly
  -d,--search-dirs [SEARCHDIRS [SEARCHDIRS ...]]
                        Module search dir
  -i,--index INDEX      Pre-generated JSON index file. If this option is used, the '-d,--search-dirs' switch is ignored.
  -w,--wrap-in-ifdef    Wrap converted lines into ifdef in host code.
  -k,--only-generate-kernels
                        Only generate kernels; do not modify host code.
  -m,--only-modify-host-code
                        Only modify host code; do not generate kernels.
  -E,--destination-dialect DESTINATIONDIALECT
                        One of: omp, hip-gpufort-rt, hip-gcc-rt
  --log-level LOGLEVEL  Set log level. Overrides config value.
  --cublas-v2           Assume cublas v2 function signatures that use a handle. Overrides config value.
  --working-dir WORKINGDIR
                        Set working directory.
  --print-config-defaults
                        Print config defaults. Config values can be overridden by providing a config file. A number of config values can be overwritten via this CLI.
  --config-file CONFIGFILE
                        Provide a config file.

```

> **NOTE**: The `--config-file` switch is parsed before
the other command line switches. If command line switches
and config parameters modify the same global `gpufort` variables,
the command line parameters overrule the config parameters.

### 2.2. Configuration files

You can configure many global variables 
that `gpufort` uses internally via a config file.
To get an overview of what you can modify and how, 
`gpufort` has a switch to print out the default values
of these variables:

```
gpufort --print-config-defaults
```

Sample output (shortened):

```python
---- gpufort -----------------------------
# logging
LOG_LEVEL                           = logging.INFO   # log level
LOG_DIR                             = "/tmp/gpufort" # directory for storing log files
LOG_DIR_CREATE                      = True           # create log dir if it does not exist
# actions to run after the command line argument parse
POST_CLI_ACTIONS                    = [] # register functions with parameters (args,unknownArgs) (see: argparse.parse_known_args)
# ...
---- translator -----------------------------
# ...
translator.FORTRAN_2_C_TYPE_MAP     = { 
  "complex" : { 
     "": FLOAT_COMPLEX,
     "16": None,
     "8": DOUBLE_COMPLEX,
     "4": FLOAT_COMPLEX,
     "2": None,
     "1": None,
  },
  #...
  "logical" : { 
    "": "bool"
  }
}
# ...
---- fort2hip -----------------------------
fort2hip.FORTRAN_MODULE_PREAMBLE    = ""
fort2hip.DEFAULT_BLOCK_SIZES        = { 1 : [128], 2 : [128,1,1], 3: [128,1,1] }
fort2hip.DEFAULT_LAUNCH_BOUNDS      = "128,1"
...
```

Users can modify these global variables directly via a config file that
they supply to the `gpufort` tool via the `--config-file CONFIGFILE`
switch. They can further append post-CLI actions and
append them to the `POST_CLI_ACTIONS` list.
Post-CLI actions are executed after all command line arguments
have been parsed. This allows to modify global
variables depending on these command line arguments.
All unrecognized arguments are further forwarded to these
actions too.

A sample config file that demonstrates modification of global variables as well as post-CLI action definition and registration is shown below:

```python
# file: myconfig.py.in 
translator.FORTRAN_2_C_TYPE_MAP["complex"]["rstd"] = "hipFloatComplex"
translator.FORTRAN_2_C_TYPE_MAP["complex"]["cstd"] = "hipDoubleComplex"

# create action
def myAction(args,unknownArgs):
    if "my_special_file.f90" in args.input.name:
        print("hello world!")

# register
POST_CLI_ACTIONS.append(myAction)
```

### 2.3. Change source and destination dialects

The Fortran source dialect**s** (CUDA Fortran, Fortran+OpenACC, ...) that `GPUFORT` should consider when parsing the translation source can be specified via a configuration file.
For the destination dialect (Fortran+OpenMP, Fortran+HIP C++, ...),
you can additionally use the command line switch:

```
-E,--destination-dialect DESTINATIONDIALECT
                        One of: omp, hip-gpufort-rt, hip-gcc-rt
```

Relevant global variables:

```
scanner.SOURCE_DIALECTS             = ["cuf","acc"] # list containing "acc" and/or "cuf"

scanner.DESTINATION_DIALECT         = "omp"         # one of: "omp", "hip-gcc-rt", or "hip-gpufort-rt"
```

### 2.4. Only modify host file or only generate kernels

When converting a code to `Fortran+C++`, you can tell `gpufort` 
to only generate kernels form the source
or to only modify the host code via the
following command line arguments:

```
-k,--only-generate-kernels
  Only generate kernels; do not modify host code.
-m,--only-modify-host-code
  Only modify host code; do not generate kernels.
```

### 2.5. Control S2S translation via GPUFORT directives

Currently `gpufort` supports only the following directives
for turning translation on and of locally

| Directive | Description |
|------|----|
| `!$gpufort on`  | Turn S2S translation on |
| `!$gpufort off` | Turn S2S translation off |

The default initial state can be configured via the config option:

```python
scanner.TRANSLATION_ENABLED_BY_DEFAULT = True
```

## 3. Debugging and optimizing GPUFORT HIP C++ kernels

HIP C++ kernel files generated with `gpufort` always
include a header file `gpufort.h`.
This header file defines a number of device functions
and various debugging routines.
The `gpufort` tool generates multiple debugging mechanisms
into the kernel launch routines that rely on the latter
routines. In this section, we discuss how you 
can debug or optimize your kernels without breaking
the existing implementation.

### 3.1. Synchronize all or only particular kernels

When generating HIP C++ kernels the `gpufort` tool generates
the following preprocessor constructs into the kernel launcher
functions:

```C++
#if defined(SYNCHRONIZE_ALL) || defined(SYNCHRONIZE_<kernel_name>)
HIP_CHECK(hipStreamSynchronize(stream));
#elif defined(SYNCHRONIZE_DEVICE_ALL) || defined(SYNCHRONIZE_DEVICE_<kernel_name>)
HIP_CHECK(hipDeviceSynchronize());
#endif
```

Therefore, you can synchronize all kernels at once or only the kernel with
name `<kernel_name>`.

### 3.2. Print input argument values

The `gpufort` tool uses the following macro to print out the values of
HIP C++ kernel arguments and launch parameters:

```C++
#define GPUFORT_PRINT_ARGS(...)
```

In the 
-generated kernel launch functions, they are guarded by the following preprocessor
construct:

```C++
#if defined(GPUFORT_PRINT_KERNEL_ARGS_ALL) || defined(GPUFORT_PRINT_KERNEL_ARGS_<kernel_name>
// ... 
#endif
```

You can thus enable argument printing for all kernels at once or only for the particular
kernel `<kernel_name>` you are interested in.

### 3.3. Print device array elements and metrics

You can use the following macros to print out device arrays:

```C++
#define GPUFORT_PRINT_ARRAY1(prefix,print_values,print_norms,A,n1,lb1) // ...
#define GPUFORT_PRINT_ARRAY2(prefix,print_values,print_norms,A,n1,n2,lb1,lb2) // ...
#define GPUFORT_PRINT_ARRAY3(prefix,print_values,print_norms,A,n1,n2,n3,lb1,lb2,lb3) // ...
// ...
#define GPUFORT_PRINT_ARRAY7(prefix,print_values,print_norms,A,n1,n2,n3,n4,n5,n6,n7,lb1,lb2,lb3,lb4,lb5,lb6,lb7) // ...
```

The device array (`A`) will be downloaded to the host and you can print out:

* all elements (line by line) via `print_values` and/or 
* a number of metrics/norms such as `l_1`, `l_2`, `sum`, `min`, `max` via `print_norms`.

The parameters `lb<i>` and `n<i>` denote the lower bound and number of elements of array dimension `<i>`, respectively.
You can further append a prefix to the front of the output in case you
want to tag it for further post-processing.

The `gpufort`tool 
matically generates such macros for input and output arrays into the

-generated kernel launchers. They are guarded by the following preprocessor construct:

```C++
#if defined(GPUFORT_PRINT_INPUT_ARRAYS_ALL) || defined(GPUFORT_PRINT_INPUT_ARRAYS_<kernel_name>)
// print arrays and norms
#elif defined(GPUFORT_PRINT_INPUT_ARRAY_NORMS_ALL) || defined(GPUFORT_PRINT_INPUT_ARRAY_NORMS_<kernel_name>)
// print only norms
#endif
```

For output arrays, the construct looks like:

```C++
#if defined(GPUFORT_PRINT_OUTPUT_ARRAYS_ALL) || defined(GPUFORT_PRINT_OUTPUT_ARRAYS_<kernel_name>)
// print arrays and norms
#elif defined(GPUFORT_PRINT_OUTPUT_ARRAY_NORMS_ALL) || defined(GPUFORT_PRINT_OUTPUT_ARRAY_NORMS_<kernel_name>)
// print only norms
#endif
```

Thus, you can enable input and output array printing for all kernels at once or only for the particular
kernel `<kernel_name>` you are interested in.

> **NOTE:** Of course, you can modify the generated code yourself too and place array printing
calls wherever you want.

### 3.4. An kernel optimizing / debugging workflow

Having the kernel extracted from the original source allows us
to debug and optimize it quite easily with the help of the `gpufort.h` debugging macros.
The principal workflow looks as follows:

1. Run the original application and print out:
   1. Input argument values of a selected kernel
   2. Input and output array elements and norms of the same kernel
2. Create a `hip.cpp` file with a `main` function.
   1. Copy the non-
 kernel launcher signature from the HIP C++ kernel file into this file and
      change it to a function call.
   2. Copy the argument types and names of the launcher to the top of the `main` function
      and change them to local variables. Remove all `const` modifiers.
   3. Duplicate the array pointers and create both a device and a host pointer `host_<array_name>`.
   4. Copy the input values you recorded previously below the local variables and convert them to assignments.
   5. Create a `hipMalloc` and `hipHostMalloc` call per host and device array after the variable assignments. Use the `n<i>` array dimension variables and `sizeof(<datatype>)` to determine a proper size.
   6. Populate the kernels on the host with application values or other values and use `hipMemcpy`to upload them to the device
   6. Use the `GPUFORT_PRINT_ARRAY` macros to print out the norms of the arrays after the launch. This way you can easily check if your optimizations changed the code. 
   7. Clean up all allocations at the end of the `main` function.

> **NOTE** Parts of this workflow can be 
mated. However, GPUFORT does not do this yet.
