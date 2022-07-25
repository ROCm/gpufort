# Scanner

This document describes the parser, tree nodes,
and output of GPUFORT's *scanner* component.
The intended readers of these notes are the developers
of the GPUFORT project. 
Be warned that this document is completely unpolished.

## Parser

The parser's input are *linemaps* obtained via GPUFORT's *linemapper* component.
Linemaps map a particular source code line to the various Fortran statements
it may contain (separated via semicolons). It further applies a preprocessor (multiply times)
that, e.g., converts singleline-if statements to multiline ones and 
evaluates C-style macros, which are often used in Fortran applications
for conditional compilation.

The parser iterates over the linemaps and then over the individual statements
per linemap. It then tokenizes the statements and tries to classify them.
Each classified statements results in a parse action that creates a tree
node or performs some other operation.

## Scanner tree nodes

Scanner tree node classes have all the prefix `ST` and share the same base class `STNode`. 
The definition files are stored in subfolder `tree`. The directory is further subdivided into
folders that contain particular nodes and code generation backends for
directive-based programming models like OpenACC (`acc`) and CUDA Fortran (`cuf`).

### OpenACC

GPUFORT records all statements--it actually records linemaps--between the start (`acc kernels`,acc parallel`, `acc serial` and loop versons of the first two) 
and end of an OpenACC compute construct (either indicated by the end of the last do loop in case `acc kernels loop` and 
`acc parallel loop` constructs, `or by the directives `acc end kernels`, `acc end serial`, `acc end parallel`, `acc end kernels loop`,
`acc end parallel loop`). GPUFORT then emits a single scanner tree node associated with the recorded code lines.
Other directives that result in runtime calls, such as `acc data`, `acc end data`, `acc enter data`, `acc exit data`,
`acc update`, and so on, are also emitted as scanner tree node.
Multiple code generation backends can be registered with the OpenACC scanner tree nodes
to enable the translation to different programming models.

GPUFORT does not create a scanner tree parent-child node hierachy for nested `acc data` regions 
and other directives placed within the start and end of such a structured data region.
Instead it creates a separate hierarchy of directives.
GPUFORT interpretes `acc kernels` regions as disguised `acc data` regions
that contain one or more compute constructs.

### OpenACC to HIP C++: Implementation details

This section explains details of the OpenACC to HIP C++
implementation.

#### Handling of structured data regions

When an `acc data` or `acc kernels` directive is encountered, this marks the begin of a structured
data region. The parser emits a tree node for the directive and 
memorizes the directive as current data or kernels directive. 
All following directives will assign this node to their `parent_directive` field but
not to their `parent` scanner tree node.
At code generation stage, a backend-specific runtime call will be written to the translation output file
that maps host data according to the specified clauses.
When the parser then encounters the matching `acc end data` or `acc end kernels`
directive that closes the data region, it reset the current `acc data` or `acc kernels` 
directive to its parent.
At code generation stage, a backend-specific runtime call will be written to the translation output file
that decrements the structured reference counters of all
data mapping associated with the data or kernels region.

In kernels regions,  we need to ensure that the structured reference counters
of explicitly mapped data are only increment once at the begin of the
region and then decremented once at th end of the kernels region.

As compute constructs within a data or kernels region have access to the
directive hierarchy, we can identify what data was already
mapped by ancestor directives.

#### Mapping global variables
~~In OpenACC, module variables can be mapped to the device via the `acc declare` directive,
which must be placed in a module's declaration list below the variable that should be mapped.
According to the OpenACC specification, the structured reference counter of such
module variables should be initialized to `1`. While fixed size arrays and scalars
should be in present in device memory from the begin of an application,
`pointers` and `allocatable` arrays should be mapped after they have been
subject to an `allocate` intrinsic call. Dynamically allocated device data should be deallocated
before the corresponding host array is subject to a `deallocate` intrinsic call.
While GPUFORT handles `pointer` and `allocatable` arrays according to the specification,
it does not statically allocate fixed size arrays but instead lazily ensures
the presence of these arrays by introducing runtime calls whereever a
module with such entities is used. This means that the first program unit that
uses the respective module, maps the `acc declare`'d fixed size arrays and scalars
to the device.~~

#### Known issues

* `no_create` is implemented incorrectly as `create` mapping.

#### Technical debt

Instead of emitting actions that decrement the structured references counters when translating an `acc end data` directive,
we should implement this directly via the OpenACC runtime. 
The latter should work with a stack for structured regions internally. 
We will pick this up in a C++ rewrite of the runtime.
