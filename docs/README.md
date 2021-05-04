# Porting Fortran OpenACC applications to Fortran + HIP with GPUFORT

## Scenario

We have an OpenACC application that can be built with 
NVIDIA's HPC toolkit.
Our aim is to port this code to port this Fortran + HIP C++.
In this report, we consider the code DYNAMICO.

## Installation of GPUFORT

GPUFORT is a python tool. It depends on a number of python packages.
Applications ported with GPUFORT have a number of dependencies
that must be installed on the host system. One
of the dependencies is HIPFORT. 

### Install latest HIPFORT

To install HIPFORT, we first need to download the latest hipfort (main or development branch).

* Main branch:

```bash
git clone git@github.com:RocmSoftwarePlatform/hipfort.git
```

* Development branch:

```bash
git clone git@github.com:RocmSoftwarePlatform/hipfort.git -b develop
```

Next we install HIPFORT to `/opt/rocm/hipfort` (recommended) via `cmake`.
As usually with `cmake`, we first create a `build` subfolder in the hipfort directory:

```bash
cd hipfort
mkdir build
```

This will result in the following HIPFORT directory structure:

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

We can test the `PATH` setting via, e.g.:

```bash
hipfc -h
```

You can test HIPFORT itself by following the steps in
https://github.com/ROCmSoftwarePlatform/hipfort/blob/master/README.md

## Install GPUFORT and OpenACC interfaces

When GPUFORT is publicly available, you can use the following
commands to donwload GPUFORT and the OpenACC runtime interfaces:

* Download GPUFORT:

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

* OpenACC interfaces and runtime for Fortran:

```bash
GIT_SSH_COMMAND='ssh -i <private_key_file> -o IdenitiesOnly=yes' git clone git@github.com:RocmSoftwarePlatform/openacc-fortran-interfaces.git
```

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

#### Download from public respositories

```bash
git clone git@github.com:RocmSoftwarePlatform/gpufort.git
```

```bash
git clone git@github.com:RocmSoftwarePlatform/openacc-fortran-interfaces.git
```

> SHOW TREES OF 

#### Download from private repositories

Until both repositories are public, we need to use the following 
git clone commands:

```bash
GIT_SSH_COMMAND='ssh -i <private_key_file> -o IdenitiesOnly=yes' git clone git@github.com:RocmSoftwarePlatform/gpufort.git
```

```bash
GIT_SSH_COMMAND='ssh -i <private_key_file> -o IdenitiesOnly=yes' git clone git@github.com:RocmSoftwarePlatform/openacc-fortran-interfaces.git
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
