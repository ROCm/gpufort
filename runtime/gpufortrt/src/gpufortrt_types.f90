! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
!> All type definitions must match those in `gpufortrt_types.h`.
module gpufortrt_types
  use iso_c_binding, only: c_int, c_size_t, c_bool, c_ptr, c_null_ptr
  private :: c_int, c_size_t, c_bool, c_ptr, c_null_ptr
  
  integer, parameter :: gpufortrt_handle_kind = c_int

  integer(gpufortrt_handle_kind), parameter :: gpufortrt_async_noval = -1 
  integer(gpufortrt_handle_kind), parameter :: gpufortrt_async_sync = -2 
  integer(gpufortrt_handle_kind), parameter :: gpufortrt_async_default = -3 
  
  !> Mapping kinds.
  enum, bind(c)
    enumerator :: gpufortrt_map_kind_undefined = 0
    enumerator :: gpufortrt_map_kind_present
    enumerator :: gpufortrt_map_kind_delete
    enumerator :: gpufortrt_map_kind_create
    enumerator :: gpufortrt_map_kind_no_create
    enumerator :: gpufortrt_map_kind_copyin
    enumerator :: gpufortrt_map_kind_copyout
    enumerator :: gpufortrt_map_kind_copy
  end enum
  
  enum, bind(c)
    enumerator :: gpufortrt_counter_none = 0
    enumerator :: gpufortrt_counter_structured
    enumerator :: gpufortrt_counter_dynamic
  end enum
  
  type, bind(c) :: gpufortrt_mapping_t
    type(c_ptr) :: hostptr   = c_null_ptr
    integer(c_size_t) :: num_bytes =  0
    integer(kind(gpufortrt_map_kind_undefined)) :: map_kind = gpufortrt_map_kind_undefined
    logical(c_bool) :: never_deallocate = .false.
  end type

contains

  subroutine gpufortrt_mapping_init(mapping,hostptr,num_bytes,map_kind,never_deallocate)
    use iso_c_binding
    implicit none
    !
    type(gpufortrt_mapping_t),intent(inout) :: mapping
    type(c_ptr),intent(in) :: hostptr
    integer(c_size_t),intent(in) :: num_bytes
    integer(kind(gpufortrt_map_kind_undefined)),intent(in) :: map_kind
    logical,intent(in),optional :: never_deallocate
    !
    logical(c_bool) :: opt_never_deallocate
    !
    opt_never_deallocate = .false._c_bool
    if ( present(never_deallocate) ) opt_never_deallocate = logical(never_deallocate,kind=c_bool)
    !
    mapping%hostptr    = hostptr
    mapping%num_bytes  = num_bytes
    mapping%map_kind   = map_kind
    mapping%never_deallocate = opt_never_deallocate
  end subroutine

end module
