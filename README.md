# GPUFORT

This project develops a source to source translation tool that is able to convert:

1. Fortran+OpenACC and CUDA Fortran -> Fortran + OpenMP 4.5+
2. Fortran+OpenACC and CUDA Fortran -> Fortran + [GCC/AOMP OpenACC/MP runtime calls] + HIP C++

The result of the first translation process, can be compiled
with AOMP, which has a Fortran frontend.
The result of the second translation process can be compiled
with hipfort or a combination of hipcc and gfortran.
Note that a OpenACC runtime is only necessary for translating
OpenACC code.

An overview of the different translation paths that we work on is shown below:

![Image](https://github.com/ROCmSoftwarePlatform/gpufort/blob/develop/gpufort.png?raw=true)

**NOTE:** GPUFORT is a research project. We made it publicly available because 
we believe that it might be helpful for some.
We want to stress that the code translation and code generation outputs produced
by GPUFORT will in most cases require manual reviewing and fixing.

## Limitations

* GPUFORT is not a compiler (yet)

GPUFORT is not intended to be a compiler.
It's main purpose is to be a translator that allows
an experienced user to fix and tune the outcomes
of the translation process. 
However, we believe GPUFORT can develop into an 
early-outlining compiler if enough effort 
is put into the project.
Given that all code and especially the grammar is
written in python3, GPUFORT can be developed at a quick 
pace.

* GPUFORT does a bad job in analyzing what code parts can be offloaded and which ones not
* GPUFORT does a bad job in reorganizing loops and assignments in order to maximize
  the available parallelism

While both would be possible as the translator works with a tree structure, 
we simply have not started to implement much in this direction yet.

* GPUFORT does not implement the full OpenACC standard (yet)

GPUFORT was developed to translate a number of HPC apps
to code formats that are well supported by AMD's ROCm ecosystem.
The development of GPUFORT is steered by the requirements
of these applications.

### Fortran-C Interoperablity Limitations

GPUFORT relies on the `iso_c_binding` interoperability mechanisms that were added to the Fortran language with 
the Fortran 2003 standard. Please be aware that the interoperability of C structs and Fortran derived types is quite limited
till this date:

* "Derived types with the C binding attribute shall not have the sequence attribute, type parameters, the extends attribute, nor type-bound procedures."
* "Every component must be of interoperable type and kind and may not have the **pointer** or **allocatable** attribute. The names of the components are irrelevant for interoperability."

(Source: https://gcc.gnu.org/onlinedocs/gfortran/Derived-Types-and-struct.html)

We are currently investigating what workarounds could be automatically applied.
Until then, you have to modify your code manually to circumvent the above limitations.

## Currently supported features:

* ACC:
   * ACC2OMP & ACC2HIP
   * Translation of data directives: `!$acc enter data`, `!$acc exit data`, `!$acc data`
   * Synchronization directives: `!$acc wait, !$acc update self/host/device`
   * Kernel and loop constructs `!$acc kernels` plus `!$acc loop` in subsequent line, `!$acc kernels loop`, `!$acc parallel` plus `!$acc loop` 
     in subsequent line, `!$acc parallel loop`, `!$acc loop`
   * Support for `!$acc routine seq` functions with scalar arguments
* CUF:
   * CUF2HIP
     * Majority of CUDA libary functionality via HIPFORT
     * Kernel and loop constructs: `!$cuf kernel do`
     * Overloaded intrinsics: `allocate`, `allocated`, `deallocate`, `deallocated`, `=`
     * Support for CUDA Fortran `attributes(global)` (array and scalar arguments), 
       and `attributes(host,device)`, `attributes(device)` procedures (only scalar arguments supported for the latter)

(List is not complete ...)

## Planned features (aka "more limitations")

* Current work focuses on:
  * ACC:
    * Initial support for `!$acc declare` (detected but not considered in codegen yet.)
    * Improve support for`!$acc parallel (loop)`
    * Add support for `!$acc parallel` without `!$acc loop` in next line)
      * Results in `gang` parallelism
    * Add support for `!$acc kernels` without `!$acc loop` in next line)
      * Auto detection of offloadable code parts 
    * Rewrite of GPUFORT Fortran runtime in (HIP) C++
  * ACC/CUF:
    * Support of derived types with allocatable, pointer members

## Installation and usage

Please take a look at the [user guide](https://bookish-adventure-5c5886a5.pages.github.io/).

## Outlook

One future goal of the project is that both translation 
processes can be mixed, which will allow users to specify what 
compute directives should be translated to HIP C++ and what compute
directives should be translated to OpenMP.

## Background

### Key ingredient: pyparsing grammars and parse actions

The fundamental ingredient of GPUFORT is its pyparsing grammar that (currently) covers a subset of the Fortran
language that plays a role in computations. This grammar is extended by additional grammar that describes
the structure of CUDA Fortran and OpenACC directives.

While easing development of a parser with shortcuts such as forward declarations and infix notation objects,
pyparsing quickly allows to generate an abstract syntax tree (AST) from a grammar with the aid 
of so-called parse actions.

A simple pyparsing grammar is shown below:

```python
import pyparsing as pp

# grammar
rvalue = pp.pyparsing_common.identifier
op   = pp.Literal("+")

expr = rvalue + op + rvalue

# test
print(expr.parseString("a + b")) # output : ['a','+','b']
```

We can directly generate an AST from the parsed string:

```python
# ...
# continue from previous snippet
class RValue():
  def __init__(self,tokens):
    self._value = tokens
class Op():
  def __init__(self,tokens):
    self._op = tokens
rvalue.setParseAction(RValue)
op.setParseAction(Op)

# run test again
print(expr.parseString("a + b")) # output : [<__main__.RValue object ...>, <__main__.Op object ...>, <__main__.RValue object ...>]
```

Now instead of the parsed strings, we have three objects in the parse result set
that describe the type of the individual tokens, i.e. we have an abstract syntax tree.
