{# SPDX-License-Identifier: MIT                                         #}
{# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved. #}
{% import "templates/gpufort_array.macros.f03" as gam %}
module gpufort_array
  use iso_c_binding
  implicit none
  
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

  ! NOTE: the below types must have exactly the
  ! same data layout as the corresponding 
  ! gpufort C/C++ structs.
{% for rank in range(1,max_rank+1) %}
{% set rank_ub = rank+1 %}
  ! {{rank}}-dimensional array
  type, bind(c) :: gpufort_array_descr{{rank}}
    type(c_ptr)       :: data_host    = c_null_ptr
    type(c_ptr)       :: data_dev     = c_null_ptr
    integer(c_int) :: num_elements = 0  !> Number of elements represented by this array.
    integer(c_int)    :: index_offset = -1 !> Offset for index calculation; scalar product of negative lower bounds and strides.
{% for d in range(1,rank_ub) %}
    integer(c_int)    :: stride{{d}}  = -1 !> Stride for dimension {{d}}
{% endfor %}
  end type
  type, bind(c) :: gpufort_array{{rank}}
    type(gpufort_array_descr{{rank}})    :: data
    integer(kind(gpufort_array_wrap_host_wrap_device)) :: alloc_mode = gpufort_array_wrap_host_alloc_device  !> Data allocation strategy. Default: 
                                                                                                             !> wrap the host and allocate device data
    integer(kind(gpufort_array_sync_none))             :: sync_mode  = gpufort_array_sync_none               !> How data should be synchronized
                                                                                                             !> during the initialization and destruction of this GPUFORT array.
    integer(c_int)                       :: num_refs               = 0                                       !> Number of references.
    integer(c_int)                       :: bytes_per_element      = -1                                      !> Bytes per data element. 
  end type{{"\n" if not loop.last}}
{% endfor %}

  ! interfaces
{{ gam.gpufort_array_fortran_interfaces(datatypes,max_rank) | indent(2,True) }}
{{ gam.gpufort_array_fortran_data_access_interfaces(datatypes,max_rank) | indent(2,True) }}

  ! subroutines
contains
{{ gam.gpufort_array_fortran_init_routines(datatypes,max_rank) | indent(2,True) }}
{{ gam.gpufort_array_fortran_wrap_routines(datatypes,max_rank) | indent(2,True) }}
{{ gam.gpufort_array_fortran_data_access_routines(datatypes,max_rank) | indent(2,True) }}
{{ gam.gpufort_array_fortran_copy_to_buffer_routines(datatypes,max_rank) | indent(2,True) }}
end module gpufort_array 
