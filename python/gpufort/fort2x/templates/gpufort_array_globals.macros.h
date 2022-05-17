{# SPDX-License-Identifier: MIT                                                 #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{########################################################################################}
{% import "gpufort.macros.h" as gm %}
{########################################################################################}
{%- macro render_defines() -%}
#ifndef GPUFORT_DEVICE_ROUTINE
#  define GPUFORT_DEVICE_ROUTINE __host__ __device__
#endif
#ifndef GPUFORT_DEVICE_ROUTINE_INLINE
#  define GPUFORT_DEVICE_ROUTINE_INLINE GPUFORT_DEVICE_ROUTINE __forceinline__
#endif
#ifndef GPUFORT_HOST_ROUTINE
#  define GPUFORT_HOST_ROUTINE __host__
#endif
{%- endmacro -%}
{########################################################################################}
{%- macro render_enums() -%}
enum class SyncMode {
  None            = 0, //> No copies between host and device data after initialization and before destruction.
  Copy            = 1, //> Copy to device after initialization and back to host before destruction.
  Copyin          = 2, //> Copy to device after initialization.
  Copyout         = 3, //> Copy from device to host before destruction.
  InvertedCopy    = 4, //> 'Copy'   in opposite direction: Copy from device to host after initialization and back to device before destruction.
  InvertedCopyin  = 5, //> 'Copyin' in opposite direction: Copy from device to host after initialization.
  InvertedCopyout = 6  //> 'Copyout'in opposite direction: Copy from host to device before destruction.
};

enum class AllocMode {
  WrapHostWrapDevice         = 0, //> Wrap host and device pointers.
  WrapHostAllocDevice        = 1, //> Wrap host pointer and allocate a new device array.
  AllocHostWrapDevice        = 2, //> Allocate new host array and wrap device pointer.
  AllocHostAllocDevice       = 3, //> Allocate new host and device arrays.
  AllocPinnedHostWrapDevice  = 4, //> Allocate new pinned host array and wrap device pointer.
  AllocPinnedHostAllocDevice = 5  //> Allocate new pinned host array and wrap device pointer.
};

enum class PrintMode {
  PrintNorms                 = 0, //> Print norms and metrics.
  PrintValues                = 1, //> Print array values; write index in Fortran (i,j,k,...) style.
  PrintValuesAndNorms        = 2, //> Print array values and norms and metrics; write index in Fortran (i,j,k,...) style.
  PrintCValues               = 3, //> Print array values; write index in C/C++ [i] style.
  PrintCValuesAndNorms       = 4  //> Print array values and norms and metrics; write index in C/C++ [i] style.
};

// Fortran/C/C++ 
enum class Direction {
  HostToHost = 0,    ///< Host-to-host operation
  HostToDevice = 1,  ///< Host-to-device operation
  DeviceToHost = 2,  ///< Device-to-host operation
  DeviceToDevice = 3 ///< Device-to-device operation
};
{%- endmacro -%}
{########################################################################################}
{%- macro render_typedefs() -%}
typedef hipStream_t queue;
typedef hipError_t error;
constexpr error success = hipSuccess;
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_globals() -%}
{{ gm.render_begin_header("GPUFORT_ARRAY_GLOBALS_H") }}
namespace gpufort {
{{ render_defines() | indent(2,True) }}
{{ render_enums() | indent(2,True) }}
{{ render_typedefs() | indent(2,True) }}
}
{{ gm.render_end_header("GPUFORT_ARRAY_GLOBALS_H") }}
{%- endmacro -%}
