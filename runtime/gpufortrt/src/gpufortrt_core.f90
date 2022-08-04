! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
module gpufortrt_core
  use iso_c_binding

  ! These routines are public

  public :: gpufortrt_ignore
  public :: gpufortrt_use_device_b
  public :: gpufortrt_copyin_b, gpufortrt_copyout_b, gpufortrt_copy_b, &
            gpufortrt_create_b, gpufortrt_no_create_b,&
            gpufortrt_present_b,&
            gpufortrt_delete_b,&
            gpufortrt_dec_struct_refs_b

  public :: gpufortrt_update_host_b, gpufortrt_update_device_b
  public :: gpufortrt_init, gpufortrt_shutdown
  public :: gpufortrt_enter_exit_data, gpufortrt_data_start, gpufortrt_data_end
  public :: gpufortrt_wait
  public :: gpufortrt_print_summary

  public :: gpufortrt_record_exists, gpufortrt_get_record_id

  public ::  &
    gpufortrt_map_kind_dec_struct_refs,&
    gpufortrt_map_kind_undefined,&
    gpufortrt_map_kind_present,&
    gpufortrt_map_kind_delete,&
    gpufortrt_map_kind_create,&
    gpufortrt_map_kind_no_create,&
    gpufortrt_map_kind_copyin,&
    gpufortrt_map_kind_copyout,&
    gpufortrt_map_kind_copy
  
  public ::  &
    gpufortrt_counter_t_none,&
    gpufortrt_counter_t_structured,&
    gpufortrt_counter_t_dynamic

  public :: gpufortrt_get_stream

  PRIVATE ! Everything below and not listed above as public is private

  !
  ! members
  !

  !> Mapping kinds.
  enum, bind(c)
    enumerator :: gpufortrt_map_kind_dec_struct_refs = -1
    enumerator :: gpufortrt_map_kind_undefined       = 0
    enumerator :: gpufortrt_map_kind_present         = 1
    enumerator :: gpufortrt_map_kind_delete          = 2
    enumerator :: gpufortrt_map_kind_create          = 3
    enumerator :: gpufortrt_map_kind_no_create       = 4
    enumerator :: gpufortrt_map_kind_copyin          = 5
    enumerator :: gpufortrt_map_kind_copyout         = 6
    enumerator :: gpufortrt_map_kind_copy            = 7
  end enum
  
  enum, bind(c)
    enumerator :: gpufortrt_counter_t_none       = 0
    enumerator :: gpufortrt_counter_t_structured = 1
    enumerator :: gpufortrt_counter_t_dynamic    = 2
  end enum

  integer, parameter :: gpufortrt_async_noval = -1 

  type, bind(c), public :: mapping_t
    type(c_ptr)       :: hostptr   = c_null_ptr
    integer(c_size_t) :: num_bytes =  0
    integer(kind(gpufortrt_map_kind_undefined)) :: map_kind = gpufortrt_map_kind_undefined
    logical           :: never_deallocate = .false.
  end type

  !> evaluate optional values
  interface eval_optval_
     module procedure :: eval_optval_l_, eval_optval_i_
  end interface

  contains

    function eval_optval_l_(optval,fallback) result(retval)
      implicit none
      logical,intent(in),optional :: optval
      logical,intent(in)          :: fallback
      logical                     :: retval
      if ( present(optval) ) then
         retval = optval
      else
         retval = fallback
      endif
    end function

    function eval_optval_i_(optval,fallback) result(retval)
      implicit none
      integer,intent(in),optional :: optval
      integer,intent(in)          :: fallback
      integer                     :: retval
      if ( present(optval) ) then
         retval = optval
      else
         retval = fallback
      endif
    end function

    !function eval_optval_3_(optval,fallback) result(retval)
    !  implicit none
    !  integer(kind(gpufortrt_map_kind_undefined)),intent(in),optional :: optval
    !  integer(kind(gpufortrt_map_kind_undefined)),intent(in)          :: fallback
    !  integer(kind(gpufortrt_map_kind_undefined))                     :: retval
    !  if ( present(optval) ) then
    !     retval = optval
    !  else
    !     retval = fallback
    !  endif
    !end function

    !
    ! debugging/analysis
    !


    ! mapping_t-bound procedures

    recursive subroutine mapping_init_(this,hostptr,num_bytes,map_kind,never_deallocate)
       use iso_c_binding
       implicit none
       !
       class(mapping_t),intent(inout) :: this
       type(c_ptr),intent(in)         :: hostptr
       integer(c_size_t),intent(in)   :: num_bytes
       integer(kind(gpufortrt_map_kind_undefined)),intent(in) :: map_kind
       logical,intent(in)             :: never_deallocate
       !
       this%hostptr    = hostptr
       this%num_bytes  = num_bytes
       this%map_kind   = map_kind
       this%never_deallocate = never_deallocate
    end subroutine

    !
    ! public
    !
    !> Ignore the result of a mapping routine.
    !> \param[in] deviceptr a device pointer.
    subroutine gpufortrt_ignore(deviceptr)
      type(c_ptr),intent(in) :: deviceptr
      !
      ! nop
    end subroutine

    ! TODO bind(c)
    subroutine gpufortrt_init()
      implicit none
    end subroutine

    ! TODO bind(c)
    subroutine gpufortrt_shutdown()
      implicit none
    end subroutine

    ! TODO bind(c) + optional arg subroutine
    subroutine gpufortrt_wait(arg,async)
      use iso_c_binding
      use hipfort
      use hipfort_check
      implicit none
      !
      integer,intent(in),optional,dimension(:) :: arg,async
      !
      integer :: i, j
      logical :: no_opt_present
      !
    end subroutine

    subroutine gpufortrt_data_start(mappings)
    !subroutine gpufortrt_data_start(device_kind,mappings)
      use iso_c_binding
      use openacc
      implicit none
      !integer,intent(in)                               :: device_kind
      type(mapping_t),dimension(:),intent(in),optional :: mappings
      !
      interface
        subroutine gpufortrt_data_start_c_impl(mappings,num_mappings) bind(c,name="gpufortrt_data_start")
          use iso_c_binding
          implicit none
          type(c_ptr),value :: mappings
          integer(c_int),value :: num_mappings
        end subroutine
      end interface 
      !
      if ( present(mappings) ) then
        call gpufortrt_data_start_c_impl(c_loc(mappings),size(mappings))  
      else
        call gpufortrt_data_start_c_impl(c_null_ptr,0)  
      endif
    end subroutine

    !> Only supposed to apply gpufortrt_map_kind_dec_struct_refs mappings.
    !> Updates the stored last record index.
    subroutine gpufortrt_data_end(mappings)
    !subroutine gpufortrt_data_start(device_kind,mappings)
      use iso_c_binding
      use openacc
      implicit none
      !integer,intent(in)                               :: device_kind
      type(mapping_t),dimension(:),intent(in),optional :: mappings
      !
      interface
          subroutine gpufortrt_data_end_c_impl(mappings,num_mappings) bind(c,name="gpufortrt_data_end")
            use iso_c_binding
            implicit none
            type(c_ptr),value :: mappings
            integer(c_int),value :: num_mappings
          end subroutine
      end interface 
      !
      if ( present(mappings) ) then
        call gpufortrt_data_end_c_impl(c_loc(mappings),size(mappings,kind=c_int))  
      else
        call gpufortrt_data_end_c_impl(c_null_ptr,0_c_int)  
      endif
  end subroutine

    subroutine gpufortrt_enter_exit_data(mappings,async,finalize)
    !subroutine gpufortrt_enter_exit_data(device_kind,mappings,async,finalize)
      use iso_c_binding
      use openacc
      implicit none
      !
      !integer,intent(in)                               :: device_kind
      type(mapping_t),dimension(:),intent(in),optional :: mappings
      integer,intent(in),optional                      :: async
      logical,intent(in),optional                      :: finalize
      !
      interface
        subroutine gpufortrt_enter_exit_data_c_impl(mappings,num_mappings,async,finalize) bind(c,name="gpufortrt_enter_exit_data")
          use iso_c_binding
          implicit none
          type(c_ptr),value,intent(in)    :: mappings
          integer(c_int),value,intent(in) :: num_mappings
          integer(c_int),value,intent(in) :: async
          logical(c_bool),value,intent(in) :: finalize
        end subroutine
      end interface 
      !
      if ( present(mappings) ) then
        call gpufortrt_enter_exit_data_c_impl(&
          c_loc(mappings),&
          size(mappings,kind=c_size_t),&
          int(eval_optval_(async,gpufortrt_async_noval),kind=c_int),&
          logical(eval_optval_(finalize,.false.),kind=c_bool)) 
      else
        call gpufortrt_enter_exit_data_c_impl(&
          c_null_ptr,&
          0_c_size_t,&
          int(eval_optval_(async,gpufortrt_async_noval),kind=c_int),&
          logical(eval_optval_(finalize,.false.),kind=c_bool)) 
      endif
    end subroutine

    !> Lookup device pointer for given host pointer.
    !> \param[in] condition condition that must be met, otherwise host pointer is returned. Defaults to '.true.'.
    !> \param[in] if_present Only return device pointer if one could be found for the host pointer.
    !>                       otherwise host pointer is returned. Defaults to '.false.'.
    !> \note Returns a c_null_ptr if the host pointer is invalid, i.e. not C associated.
    function gpufortrt_use_device_b(hostptr,num_bytes,condition,if_present) result(resultptr)
      use iso_fortran_env
      use iso_c_binding
      implicit none
      type(c_ptr),intent(in)       :: hostptr
      integer(c_size_t),intent(in) :: num_bytes
      logical(c_bool),intent(in),optional  :: condition, if_present
      !
      type(c_ptr) :: resultptr
      !
      interface
        subroutine gpufortrt_use_device_b(
            type(c_ptr),intent(in)       :: hostptr
            integer(c_size_t),intent(in) :: num_bytes
            logical(c_bool),intent(in),optional  :: condition, if_present
      end interface
    end function

    function gpufortrt_present_b(hostptr,num_bytes,update_struct_refs) result(deviceptr)
      use iso_fortran_env
      use iso_c_binding
      implicit none
      type(c_ptr),intent(in)       :: hostptr
      integer(c_size_t),intent(in) :: num_bytes
      logical,intent(in),optional  :: update_struct_refs
      !
      type(c_ptr) :: deviceptr
      !
    end function

    subroutine gpufortrt_dec_struct_refs_b(hostptr,async)
      use iso_fortran_env
      use iso_c_binding
      implicit none
      type(c_ptr),intent(in)      :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
    end subroutine

    !> no_create( list ) parallel, kernels, serial, data
    !> When entering the region, if the data in list is already present on
    !> the current device, the structured reference count is incremented
    !> and that copy is used. Otherwise, no action is performed and any
    !> device code in the construct will use the local memory address
    !> for that data.
    !>
    !> \note We just return the device pointer here (or c_null_ptr) and do not modify the counters.
    function gpufortrt_no_create_b(hostptr) result(deviceptr)
      use iso_c_binding
      implicit none
      type(c_ptr), intent(in)     :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      logical :: success
      integer :: loc
      !
      if ( .not. initialized_ ) call gpufortrt_init()
      if ( .not. c_associated(hostptr) ) then
        ERROR STOP "gpufortrt_no_create_b: hostptr not c_associated"
      else
        loc = record_list_find_record_(hostptr,success)
        deviceptr = c_null_ptr
        if ( success ) deviceptr = record_list_%records(loc)%deviceptr
      endif
    end function

    !> create( list ) parallel, kernels, serial, data, enter data,
    !> declare
    !> When entering the region or at an enter data directive,
    !> if the data in list is already present on the current device, the
    !> appropriate reference count is incremented and that copy
    !> is used. Otherwise, it allocates device memory and sets the
    !> appropriate reference count to one. When exiting the region,
    !> the structured reference count is decremented. If both reference
    !> counts are zero, the device memory is deallocated.
    !>
    !> \note Only use this when entering not when exiting
    function gpufortrt_create_b(hostptr,num_bytes,async,never_deallocate,&
                                update_struct_refs,update_dyn_refs) result(deviceptr)
      use iso_c_binding
      implicit none
      type(c_ptr),intent(in)       :: hostptr
      integer(c_size_t),intent(in) :: num_bytes
      integer,intent(in),optional  :: async ! ignored as there is no hipMallocAsync
      logical,intent(in),optional  :: never_deallocate
      logical,intent(in),optional  :: update_struct_refs,update_dyn_refs
      !
      type(c_ptr) :: deviceptr
      !
      integer :: loc
      !
      if ( .not. initialized_ ) call gpufortrt_init()
      if ( .not. c_associated(hostptr) ) then
        ERROR STOP "gpufortrt_create_b: hostptr not c_associated"
      else
        loc       = record_list_use_increment_record_(hostptr,num_bytes,&
                      gpufortrt_map_kind_create,async,never_deallocate,&
                      update_struct_refs,update_dyn_refs)
        deviceptr = record_list_%records(loc)%deviceptr
      endif
    end function

    !> The delete clause may appear on exit data directives.
    !>
    !> For each var in varlist, if var is in shared memory, no action is taken; if var is not in shared memory,
    !> the delete clause behaves as follows:
    !> If var is not present in the current device memory, a runtime error is issued.
    !> Otherwise, the dynamic reference counter is updated:
    !> On an exit data directive with a finalize clause, the dynamic reference counter
    !> is set to zero.
    !> Otherwise, a present decrement action with the dynamic reference counter is performed.
    !> If var is a pointer reference, a detach action is performed. If both structured and dynamic
    !> reference counters are zero, a delete action is performed.
    !> An exit data directive with a delete clause and with or without a finalize clause is
    !> functionally equivalent to a call to
    !> the acc_delete_finalize or acc_delete API routine, respectively, as described in Section 3.2.23.
    !>
    !> \note use
    subroutine gpufortrt_delete_b(hostptr,finalize)
      use iso_c_binding
      implicit none
      type(c_ptr), intent(in)     :: hostptr
      logical,intent(in),optional :: finalize
      !
      logical :: success, opt_finalize
      integer :: loc
      !
      if ( .not. initialized_ ) ERROR STOP "gpufortrt_delete_b: runtime not initialized"
      if ( .not. c_associated(hostptr) ) then
#ifndef NULLPTR_MEANS_NOOP
        ERROR STOP "gpufortrt_delete_b: hostptr not c_associated"
#endif
        return
      else
        opt_finalize = .false.
        if ( present(finalize) ) opt_finalize = finalize
        if ( opt_finalize ) then
          loc = record_list_find_record_(hostptr,success)
          if ( .not. success ) ERROR STOP "gpufortrt_delete_b: no record found for hostptr"
          if ( c_associated(record_list_%records(loc)%hostptr,hostptr) ) then
            call record_release_(record_list_%records(loc))
          else
            ERROR STOP "gpufortrt_delete_b: called on subsection of mapped data"
          endif
        else
          call record_list_decrement_release_record_(hostptr,&
            update_dyn_refs=.true.,&
            veto_copy_to_host=.true.)
        endif
      endif
    end subroutine

    !> copyin( list ) parallel, kernels, serial, data, enter data,
    !> declare
    !>
    !> When entering the region or at an enter data directive,
    !> if the data in list is already present on the current device, the
    !> appropriate reference count is incremented and that copy is
    !> used. Otherwise, it allocates device memory and copies the
    !> values from the encountering thread and sets the appropriate
    !> reference count to one.
    !>
    !> When exiting the region the structured
    !> reference count is decremented. If both reference counts are
    !> zero, the device memory is deallocated.
    !>
    !> \note Exiting case handled in gpufortrt_exit_region
    function gpufortrt_copyin_b(hostptr,num_bytes,async,never_deallocate,&
        update_struct_refs,update_dyn_refs) result(deviceptr)
      use iso_c_binding
      implicit none
      type(c_ptr), intent(in)      :: hostptr
      integer(c_size_t),intent(in) :: num_bytes
      integer,intent(in),optional  :: async
      logical,intent(in),optional  :: never_deallocate
      logical,intent(in),optional  :: update_struct_refs,update_dyn_refs
      !
      type(c_ptr) :: deviceptr
      !
      integer :: loc
      !
      if ( .not. initialized_ ) call gpufortrt_init()
      if ( .not. c_associated(hostptr) ) then
#ifndef NULLPTR_MEANS_NOOP
        ERROR STOP "gpufortrt_copyin_b: hostptr not c_associated"
#endif
        deviceptr = c_null_ptr
      else
        loc       = record_list_use_increment_record_(hostptr,num_bytes,gpufortrt_map_kind_copyin,async,never_deallocate)
        deviceptr = record_list_%records(loc)%deviceptr
      endif
    end function

    !> copyout( list ) parallel, kernels, serial, data, exit data,
    !> declare
    !> When entering the region, if the data in list is already present on
    !> the current device, the structured reference count is incremented
    !> and that copy is used. Otherwise, it allocates device memory and
    !> sets the structured reference count to one.
    !>
    !> At an exit data
    !> directive with no finalize clause or when exiting the region,
    !> the appropriate reference count is decremented. At an exit
    !> data directive with a finalize clause, the dynamic reference
    !> count is set to zero. In any case, if both reference counts are zero,
    !> the data is copied from device memory to the encountering
    !> thread and the device memory is deallocated.
    !>
    !> \note Only returns valid device pointer if none of the counters
    !> is updated, which implies that
    function gpufortrt_copyout_b(hostptr,num_bytes,async,&
            update_struct_refs,update_dyn_refs) result(deviceptr)
      use iso_c_binding
      implicit none
      ! Return type(c_ptr)
      type(c_ptr), intent(in)      :: hostptr
      integer(c_size_t),intent(in) :: num_bytes
      integer,intent(in),optional  :: async
      logical,intent(in),optional  :: update_struct_refs,update_dyn_refs
      !
      type(c_ptr) :: deviceptr
      !
      integer :: loc
      !
      if ( .not. initialized_ ) call gpufortrt_init()
      if ( .not. c_associated(hostptr) ) then
#ifndef NULLPTR_MEANS_NOOP
        ERROR STOP "gpufortrt_copyout_b: hostptr not c_associated"
#endif
        deviceptr = c_null_ptr
      else
        if ( eval_optval_(update_struct_refs,.false.) ) then
            ! clause attached to data directive
            loc       = record_list_use_increment_record_(hostptr,num_bytes,&
                          gpufortrt_map_kind_copyout,async,.false.,&
                          update_struct_refs,update_dyn_refs)
            deviceptr = c_null_ptr ! not needed in this case
        else if ( eval_optval_(update_dyn_refs,.false.) ) then
            ! clause attached to exit data directive
            call record_list_decrement_release_record_(&
                hostptr,.false.,.false.,.false.)
            deviceptr = c_null_ptr ! not needed in this case
        else
            ! clause directly associated with compute-construct
            loc       = record_list_use_increment_record_(hostptr,num_bytes,&
                          gpufortrt_map_kind_copyout,async,.false.,&
                          .false.,.false.)
            deviceptr = record_list_%records(loc)%deviceptr
        endif
      endif
    end function

    !> copy( list ) parallel, kernels, serial, data, declare
    !> When entering the region, if the data in list is already present on
    !> the current device, the structured reference count is incremented
    !> and that copy is used. Otherwise, it allocates device memory
    !> and copies the values from the encountering thread and sets
    !> the structured reference count to one. When exiting the region,
    !> the structured reference count is decremented. If both reference
    !> counts are zero, the data is copied from device memory to the
    !> encountering thread and the device memory is deallocated.
    !>
    !> \note Exiting case handled in gpufortrt_exit_region
    function gpufortrt_copy_b(hostptr,num_bytes,async,&
            update_struct_refs) result(deviceptr)
      use iso_c_binding
      implicit none
      ! Return type(c_ptr)
      type(c_ptr), intent(in)      :: hostptr
      integer(c_size_t),intent(in) :: num_bytes
      integer,intent(in),optional  :: async
      logical,intent(in),optional  :: update_struct_refs
      !
      type(c_ptr) :: deviceptr
      !
      integer :: loc
      !
      if ( .not. initialized_ ) call gpufortrt_init()
      if ( .not. c_associated(hostptr) ) then
#ifndef NULLPTR_MEANS_NOOP
        ERROR STOP "gpufortrt_copy_b: hostptr not c_associated"
#endif
        deviceptr = c_null_ptr
      else
        loc       = record_list_use_increment_record_(hostptr,num_bytes,&
                      gpufortrt_map_kind_copy,&
                      async,.false.,&
                      update_struct_refs,.false.)
        deviceptr = record_list_%records(loc)%deviceptr
      endif
    end function

    !> Update Directive
    !>
    !> The update directive copies data between the memory for the
    !> encountering thread and the device. An update directive may
    !> appear in any data region, including an implicit data region.
    !>
    !> FORTRAN
    !>
    !> !$acc update [clause [[,] clause]…]
    !>
    !> CLAUSES
    !>
    !> self( list ) or host( list )
    !>   Copies the data in list from the device to the encountering
    !>   thread.
    !> device( list )
    !>   Copies the data in list from the encountering thread to the
    !>   device.
    subroutine gpufortrt_update_host_b(hostptr,condition,if_present,async)
      use iso_c_binding
      implicit none
      type(c_ptr),intent(in)      :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      integer :: loc
      !
      logical :: success, opt_condition, opt_if_present
      !
      if ( .not. initialized_ ) ERROR STOP "gpufortrt_update_host_b: runtime not initialized"
      if ( .not. c_associated(hostptr) ) then
#ifndef NULLPTR_MEANS_NOOP
        ERROR STOP "gpufortrt_update_host_b: hostptr not c_associated"
#endif
        return
      endif
      opt_condition  = .true.
      opt_if_present = .false.
      if ( present(condition) )  opt_condition  = condition
      if ( present(if_present) ) opt_if_present = if_present
      !
      if ( opt_condition ) then
        loc = record_list_find_record_(hostptr,success)
        !
        if ( .not. success .and. .not. opt_if_present ) ERROR STOP "gpufortrt_update_host_b: no deviceptr found for hostptr"
        !
        if ( success .and. present(async) ) then
          call record_copy_to_host_(record_list_%records(loc),async)
        else if ( success ) then
          call record_copy_to_host_(record_list_%records(loc))
        endif
      endif
    end subroutine

    !> Update Directive
    !>
    !> The update directive copies data between the memory for the
    !> encountering thread and the device. An update directive may
    !> appear in any data region, including an implicit data region.
    !>
    !> FORTRAN
    !>
    !> !$acc update [clause [[,] clause]…]
    !>
    !> CLAUSES
    !>
    !> self( list ) or host( list )
    !>   Copies the data in list from the device to the encountering
    !>   thread.
    !> device( list )
    !>   Copies the data in list from the encountering thread to the
    !>   device.
    !> if( condition )
    !>   When the condition is zero or .false., no data will be moved to
    !>   or from the device.
    !> if_present
    !>   Issue no error when the data is not present on the device.
    !> async [( expression )]
    !>   The data movement will execute asynchronously with the
    !>   encountering thread on the corresponding async queue.
    !> wait [( expression-list )] - TODO not implemented
    !>   The data movement will not begin execution until all actions on
    !>   the corresponding async queue(s) are complete.
    subroutine gpufortrt_update_device_b(hostptr,condition,if_present,async)
      use iso_c_binding
      implicit none
      type(c_ptr),intent(in)      :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      integer :: loc
      logical :: success, opt_condition, opt_if_present
      !
      if ( .not. initialized_ ) ERROR STOP "gpufortrt_update_device_b: runtime not initialized"
      if ( .not. c_associated(hostptr) ) then
#ifndef NULLPTR_MEANS_NOOP
        ERROR STOP "gpufortrt_update_device_b: deviceptr not c_associated"
#endif
        return
      endif
      opt_condition  = .true.
      opt_if_present = .false.
      if ( present(condition) )  opt_condition  = condition
      if ( present(if_present) ) opt_if_present = if_present
      !
      if ( opt_condition ) then
        loc = record_list_find_record_(hostptr,success)
        !
        if ( .not. success .and. .not. opt_if_present ) ERROR STOP "gpufortrt_update_device_b: no deviceptr found for hostptr"
        !
        if ( success .and. present(async) ) then
          call record_copy_to_device_(record_list_%records(loc),async)
        else if ( success ) then
          call record_copy_to_device_(record_list_%records(loc))
        endif
      endif
    end subroutine

    function gpufortrt_get_stream(queue_id) result (stream)
       implicit none
       integer, intent(in) :: queue_id
       !
       type(c_ptr) :: stream
       !
       if ( queue_id .eq. 0 ) then
          stream = c_null_ptr
       else
          call ensure_queue_exists_(queue_id)
          stream = queues_(queue_id)%queueptr
       endif
    end function
end module
