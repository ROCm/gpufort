{# SPDX-License-Identifier: MIT                                         #}
{# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved. #}
{% import "templates/gpufort_arrays.macros.f03" as gam %}
module gpufort_arrays
  use iso_c_binding
  ! NOTE: the below types must have exactly the
  ! same data layout as the corresponding 
  ! gpufort C/C++ structs.

{% for rank in range(1,max_rank+1) %}
{% set rank_ub = rank+1 %}
  ! {{rank}}-dimensional array
  type, bind(c) :: gpufort_gpu_array_{{rank}}
    type(c_ptr)       :: data_host    = c_null_ptr;
    type(c_ptr)       :: data_dev     = c_null_ptr;
    integer(c_size_t) :: num_elements = 0;  !> Number of represented by this array.
    integer(c_int)    :: index_offset = -1; !> Offset for index calculation; scalar product of negative lower bounds and strides.
{% for d in range(1,rank_ub) %}
    integer(c_int)    :: stride{{d}}  = -1; !> Stride for dimension {{d}}
{% endfor %}
  end type
  type, bind(c) :: gpufort_mapped_array_{{rank}}
    type(gpufort_gpu_array_{{rank}}) :: data
    logical(c_bool)                  :: pinned                 = .false.; !> If the host data is pinned. 
    logical(c_bool)                  :: copyout_at_destruction = .false.; !> If the device data should be copied back to the host when this struct is destroyed.
    logical(c_bool)                  :: owns_host_data         = .false.; !> If this is only a wrapper, i.e. no memory management is performed.
    logical(c_bool)                  :: owns_device_data       = .false.; !> If this is only a wrapper, i.e. no memory management is performed.
    integer(c_int)                   :: num_refs               = 0;       !> Number of references.
    integer(c_int)                   :: bytes_per_element      = -1;      !> Bytes per data element. 
  end type{{"\n" if not loop.last}}
{% endfor %}

  ! interfaces
{{ gam.gpufort_arrays_fortran_interfaces(datatypes,max_rank) | indent(2,True) }}

  ! subroutines
contains
{{ gam.gpufort_arrays_fortran_routines(datatypes,max_rank) | indent(2,True) }}
end module gpufort_arrays 
