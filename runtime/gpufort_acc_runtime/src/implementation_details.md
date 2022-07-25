# GPUFORT OpenACC Runtime

This document describes implementation details
of GPUFORT's OpenACC runtime.

## Source files

Please do not edit file `gpufort_acc_runtime.f90` manually as this
file is generated from the template `gpufort_acc_runtime.template.f90`.
`gpufort_acc_runtime` is a frontend for `gpufort_acc_runtime_base.f90`,
which implements the fundamental data structures and routines
of the runtime.

## Mapping procedures

Per default, the mapping procedures (`gpufort_acc_<clause-name>`, `<clause-name` is one of `present`, `create`, ... ) do not increment 
the structured or dynamic reference counters of a record. 
This has to specified via the optional arguments `update_struct_refs` and `update_dyn_refs`.
The runtime calls for starting and ending structured and unstructured
data regions (`gpufort_acc_data_start`, `gpufort_acc_data_end`, and `gpufort_acc_enter_exit_data`)
use these optional arguments.
Exceptions to the aforementioned rule, are the `gpufort_acc_delete` and `gpufort_decrement_struct_refs` 
mapping procedures that always decrement the dynamic and structured
reference counters, respectively.
