{# SPDX-License-Identifier: MIT                                                 #}
{# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved. #}
{########################################################################################}
{%- macro render_enums() -%}
enum, bind(c)
  enumerator :: gpufort_array_sync_none             = 0 !> No copies between host and device data after initialization and before destruction.
  enumerator :: gpufort_array_sync_copy             = 1 !> Copy to device after initialization and back to host before destruction.
  enumerator :: gpufort_array_sync_copyin           = 2 !> Copy to device after initialization.
  enumerator :: gpufort_array_sync_copyout          = 3 !> Copy back to host before destruction.
  enumerator :: gpufort_array_sync_inverted_copy    = 4 !> 'Copy'   in opposite direction: Copy from device to host after initialization and back to device before destruction.
  enumerator :: gpufort_array_sync_inverted_copyin  = 5 !> 'Copyin' in opposite direction: Copy from device to host after initialization.
  enumerator :: gpufort_array_sync_inverted_copyout = 6 !> 'Copyout'in opposite direction: Copy from host to device before destruction.
end enum

enum, bind(c)
  enumerator :: gpufort_array_wrap_host_wrap_device          = 0 !> Wrap host and device pointers.
  enumerator :: gpufort_array_wrap_host_alloc_device         = 1 !> Wrap host pointer and allocate a new device array.
  enumerator :: gpufort_array_alloc_host_wrap_device         = 2 !> Allocate new host array and wrap device pointer.
  enumerator :: gpufort_array_alloc_host_alloc_device        = 3 !> Allocate new host and device arrays.
  enumerator :: gpufort_array_alloc_pinned_host_wrap_device  = 4 !> Allocate new pinned host array and wrap device pointer.
  enumerator :: gpufort_array_alloc_pinned_host_alloc_device = 5 !> Allocate new pinned host array and wrap device pointer.
end enum
{%- endmacro -%}
{########################################################################################}
{%- macro render_typedefs() -%}
integer,parameter :: gpufort_error_kind = kind(hipSuccess)
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_globals_module() -%}
module gpufort_array_globals
  use iso_c_binding
  use hipfort_enums
  implicit none
{{ render_enums() | indent(2,True) }}
{{ render_typedefs() | indent(2,True) }}
end module
{%- endmacro -%}
