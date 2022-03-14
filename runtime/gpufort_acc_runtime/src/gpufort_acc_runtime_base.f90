! SPDX-License-Identifier: MIT                                                
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "gpufort_acc_runtime.def"

#define blocked_size(num_bytes) ((((num_bytes)+BLOCK_SIZE-1)/BLOCK_SIZE) * BLOCK_SIZE)

module gpufort_acc_runtime_base
  use iso_c_binding

  ! These routines are public

  public :: gpufort_acc_ignore
  public :: gpufort_acc_copyin_b, gpufort_acc_copyout_b, gpufort_acc_copy_b, & 
            gpufort_acc_create_b, gpufort_acc_no_create_b, 
            gpufort_acc_present_b, &
            gpufort_acc_present_or_create_b, &
            gpufort_acc_present_or_copyin_b, gpufort_acc_present_or_copyout_b, gpufort_acc_present_or_copy_b, & 
            gpufort_acc_delete_b 
  public :: gpufort_acc_update_host_b, gpufort_acc_update_device_b
  public :: gpufort_acc_init, gpufort_acc_shutdown
  public :: gpufort_acc_enter_region, gpufort_acc_exit_region
  public :: gpufort_acc_wait
  public :: gpufort_acc_runtime_print_summary
  
  public :: gpufort_acc_runtime_record_exists, gpufort_acc_runtime_get_record_id
    
  public :: gpufort_acc_event_undefined,&
            gpufort_acc_event_create,&
            gpufort_acc_event_copyin,&
            gpufort_acc_event_copyout,&
            gpufort_acc_event_copy     

  public :: gpufort_acc_get_stream

  PRIVATE ! Everything below and not listed above as public is private

  !
  ! members
  !
  
  integer, save :: LOG_LEVEL                = 0
  integer, save :: MAX_QUEUES               = 64
  integer, save :: INITIAL_RECORDS_CAPACITY = 4096
  ! reuse/fragmentation controls
  integer, save :: BLOCK_SIZE             = 32
  real, save    :: REUSE_THRESHOLD        = 0.9 ! only reuse record if mem_new>=factor*mem_old 
  integer, save :: NUM_REFS_TO_DEALLOCATE = -5  ! dealloc device mem only if num_refs takes this value 

  integer, save :: record_creation_counter_  = 0
  logical, save :: initialized_              = .false.
  integer, save :: last_queue_index_         = 0
 
  !> Creational events for a record
  enum, bind(c) 
    enumerator :: gpufort_acc_event_undefined = 0 
    enumerator :: gpufort_acc_event_create    = 1 
    enumerator :: gpufort_acc_event_copyin    = 2 
    enumerator :: gpufort_acc_event_copyout   = 3 
    enumerator :: gpufort_acc_event_copy      = 4 
  end enum 

  !> Data structure that maps a host to a device pointer.
  type :: t_record
    integer           :: id             = -1
    type(c_ptr)       :: hostptr        = c_null_ptr
    type(c_ptr)       :: deviceptr      = c_null_ptr
    integer           :: region         = -1
    integer(c_size_t) :: num_bytes      = 0
    integer(c_size_t) :: num_bytes_used = 0
    integer           :: num_refs       = 0
    integer(kind(gpufort_acc_event_create)) :: creational_event = gpufort_acc_event_undefined

    contains 
      procedure :: print              => t_record_print_
      procedure :: is_initialized     => t_record_is_initialized_
      procedure :: is_used            => t_record_is_used_
      procedure :: is_released        => t_record_is_released_
      procedure :: is_subarray        => t_record_is_subarray_
      procedure :: setup              => t_record_setup_
      procedure :: release            => t_record_release_
      procedure :: destroy            => t_record_destroy_
      procedure :: copy_to_device     => t_record_copy_to_device_
      procedure :: copy_to_host       => t_record_copy_to_host_
      procedure :: decrement_num_refs => t_record_decrement_num_refs_
      procedure :: increment_num_refs => t_record_increment_num_refs_
  end type

  !> Data structure managing records and storing associated metadata.
  type :: t_record_list
    type(t_record), allocatable :: records(:)
    integer                     :: last_record_index  = 0
    integer                     :: current_region     = 0
    integer(c_size_t)           :: total_memory_bytes = 0
    
    contains   
      procedure :: is_initialized           => t_record_list_is_initialized_
      procedure :: initialize               => t_record_list_initialize_
      procedure :: grow                     => t_record_list_grow_
      procedure :: destroy                  => t_record_list_destroy_
      procedure :: find_record              => t_record_list_find_record_
      procedure :: find_available_record    => t_record_list_find_available_record_
      procedure :: use_increment_record     => t_record_list_use_increment_record_
      procedure :: decrement_release_record => t_record_list_decrement_release_record_
  end type

  !> Data structure to map integer numbers to queues.  
  type :: t_queue
    type(c_ptr) :: queueptr = c_null_ptr
    
    contains 
      procedure :: is_initialized     => t_queue_is_initialized_
      procedure :: initialize         => t_queue_initialize_
      procedure :: destroy            => t_queue_destroy_
  end type

  type(t_record_list),save       :: record_list_
  type(t_queue),allocatable,save :: queues_(:)

  !> evaluate optional values
  interface eval_optval_
     module procedure :: eval_optval_1_, eval_optval_2_
  end interface 

  contains

    function eval_optval_1_(optval,fallback) result(retval)
      implicit none
      logical,optional,intent(in) :: optval
      logical,intent(in)          :: fallback
      logical                     :: retval
      if ( present(optval) ) then
         retval = optval       
      else
         retval = fallback
      endif
    end function
    
    function eval_optval_2_(optval,fallback) result(retval)
      implicit none
      integer,optional,intent(in) :: optval
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
    !  integer(kind(gpufort_acc_event_undefined)),optional,intent(in) :: optval
    !  integer(kind(gpufort_acc_event_undefined)),intent(in)          :: fallback
    !  integer(kind(gpufort_acc_event_undefined))                     :: retval
    !  if ( present(optval) ) then
    !     retval = optval       
    !  else
    !     retval = fallback
    !  endif
    !end function
    
    !
    ! debugging/analysis
    !
  
    function gpufort_acc_runtime_record_exists(hostptr,id,print_record) result(success)
      use iso_fortran_env
      use iso_c_binding 
      implicit none
      type(c_ptr),intent(in),optional :: hostptr
      integer,intent(in),optional     :: id
      logical,intent(in),optional     :: print_record
      !
      logical :: success
      !
      integer     :: i
      logical     :: opt_print_record
      type(c_ptr) :: opt_hostptr
      integer     :: opt_id
      !
      opt_print_record = .false.
      opt_hostptr      = c_null_ptr 
      opt_id           = -1
      !
      if ( present(print_record) ) opt_print_record = print_record
      if ( present(hostptr) )      opt_hostptr = hostptr
      if ( present(id) )           opt_id      = id
      !
      success = .false.
      do i = 1, record_list_%last_record_index
        
         if ( ( record_list_%records(i)%is_subarray(opt_hostptr,0_8) ) .or. &
              ( record_list_%records(i)%id .eq. opt_id ) ) then
          success = .true.
          if ( opt_print_record ) then 
            CALL record_list_%records(i)%print() 
            flush(output_unit)
          endif
          exit ! loop
        endif 
      end do
      if ( .not. success .and. opt_print_record ) ERROR STOP "no record found"
    end function        
   
    function gpufort_acc_runtime_get_record_id(hostptr) result(id)
      use iso_c_binding 
      implicit none
      type(c_ptr),intent(in) :: hostptr
      !
      integer :: id
      !
      integer :: i
      !
      do i = 1, record_list_%last_record_index
        if ( record_list_%records(i)%is_subarray(hostptr,0_8) ) then
          id = record_list_%records(i)%id
          exit ! loop
        endif 
      end do
    end function
   
    subroutine gpufort_acc_runtime_print_summary(print_records)
      use iso_fortran_env
      use iso_c_binding
      use hipfort
      use hipfort_check
      implicit none
      logical,intent(in),optional :: print_records
      !
      integer :: i, j
      integer(c_size_t) :: free_memory, total_memory 
      logical           :: opt_print_records
      !
      opt_print_records = .false.
      !
      if (  present(print_records) ) opt_print_records = print_records
      !
      CALL hipCheck(hipMemGetInfo(free_memory,total_memory))
      !
      print *, "SUMMARY"
      print *, ""
      print *, "stats:"
      print *, "- total records created       = ", record_creation_counter_
      print *, "- total memory allocated (B)  = ", total_memory
      print *, "- HIP used memory        (B)  = ", (total_memory - free_memory), "/", total_memory
      print *, "- current region              = ", record_list_%current_region
      print *, "- last record index           = ", record_list_%last_record_index
      print *, "- device memory allocated (B) = ", record_list_%total_memory_bytes
      print *, ""
      if ( opt_print_records ) then
        do i = 1, record_list_%last_record_index
           print *, "- record ", i, ":"
           CALL record_list_%records(i)%print()
           flush(output_unit)
        end do
      endif
    end subroutine

    !
    ! t_queue-bound procedures
    !
    
    function t_queue_is_initialized_(queue) result(ret)
      use iso_c_binding
      implicit none
      class(t_queue),intent(in) :: queue
      logical(c_bool) :: ret
      !
      ret = c_associated( queue%queueptr )
    end function
    
    subroutine t_queue_initialize_(queue) 
      use hipfort
      use hipfort_check
      implicit none
      class(t_queue),intent(inout) :: queue
      !
      call hipCheck(hipStreamCreate(queue%queueptr))
    end subroutine
    
    subroutine t_queue_destroy_(queue)
      use hipfort
      use hipfort_check
      implicit none
      class(t_queue),intent(inout) :: queue
      !
      call hipCheck(hipStreamDestroy(queue%queueptr))
      queue%queueptr = c_null_ptr
    end subroutine
    
    !
    ! queues_
    !
    
    !> \note Not thread safe 
    subroutine ensure_queue_exists_(id)
       implicit none
       integer, intent(in) :: id
       !
       if ( id .gt. 0 .and. id .le. MAX_QUEUES ) then
          if ( .not. queues_(id)%is_initialized() ) then
            call queues_(id)%initialize()
          endif
          last_queue_index_ = max(id, last_queue_index_)
       else if ( id .gt. MAX_QUEUES ) then
         ERROR STOP "gpufort_acc_runtime: ensure_queue_exists_: queue id greater than parameter MAX_QUEUES" 
       else
         ERROR STOP "gpufort_acc_runtime: ensure_queue_exists_: queue id must be greater than 0" 
       endif
    end subroutine
   
    !> \note Not thread safe 
    subroutine destroy_queue_(id)
       implicit none
       integer, intent(in) :: id
       !
       if ( id .gt. 0 .and. id .le. MAX_QUEUES ) then
          if ( queues_(id)%is_initialized() ) then
            call queues_(id)%destroy()
          endif
          if ( id .eq. last_queue_index_ ) then
            last_queue_index_ = last_queue_index_ - 1
          endif
       else if ( id .gt. MAX_QUEUES ) then
         ERROR STOP "gpufort_acc_runtime: destroy_queue_: queue id greater than parameter MAX_QUEUES"
       else
         ERROR STOP "gpufort_acc_runtime: destroy_queue_: queue id must be greater than 0" 
       endif
    end subroutine
    
    !
    ! t_record-bound procedures
    !
    
    subroutine t_record_print_(record)
      use iso_c_binding
      use gpufort_acc_runtime_c_bindings
      implicit none
      class(t_record),intent(in) :: record
      !
      CALL print_record(&
        record%id,&
        record%is_initialized(),&
        record%hostptr,&
        record%deviceptr,&
        record%num_bytes,&
        record%num_refs,&
        record%region,&
        record%creational_event)
    end subroutine
    
    function t_record_is_initialized_(record) result(ret)
      use iso_c_binding
      implicit none
      class(t_record),intent(in) :: record
      logical(c_bool) :: ret
      !
      ret = c_associated(record%deviceptr)
    end function
    
    function t_record_is_used_(record) result(ret)
      use iso_c_binding
      implicit none
      class(t_record),intent(in) :: record
      logical(c_bool) :: ret
      !
      ret = record%is_initialized() .and. &
            record%num_refs > 0
    end function
    
    function t_record_is_released_(record) result(ret)
      use iso_c_binding
      implicit none
      class(t_record),intent(in) :: record
      logical(c_bool) :: ret
      !
      ret = record%is_initialized() .and. &
            record%num_refs <= 0
    end function
    
   subroutine t_record_copy_to_device_(record,async)
      use iso_c_binding
      use hipfort_check
      use hipfort_hipmemcpy
      implicit none
      class(t_record),intent(in)  :: record
      integer,intent(in),optional :: async
      !
#ifndef BLOCKING_COPIES
      if ( present(async) ) then
        if ( async .gt. 0 ) then
          call ensure_queue_exists_(async)
          call hipCheck(hipMemcpyAsync(record%deviceptr,record%hostptr,&
                  record%num_bytes_used,hipMemcpyHostToDevice,queues_(async)%queueptr))
        else
          call hipCheck(hipMemcpyAsync(record%deviceptr,record%hostptr,&
                  record%num_bytes_used,hipMemcpyHostToDevice,c_null_ptr))
        endif
      else
#endif
        call hipCheck(hipMemcpy(record%deviceptr,record%hostptr,record%num_bytes_used,hipMemcpyHostToDevice))
#ifndef BLOCKING_COPIES
      endif
#endif
    end subroutine
    
    subroutine t_record_copy_to_host_(record,async)
      use iso_c_binding
      use hipfort_check
      use hipfort_hipmemcpy
      implicit none
      class(t_record),intent(in) :: record
      integer,intent(in),optional :: async
      !
#ifndef BLOCKING_COPIES
      if ( present(async) ) then
        if ( async .gt. 0 ) then
          call ensure_queue_exists_(async)
          call hipCheck(hipMemcpyAsync(record%hostptr,record%deviceptr,&
                  record%num_bytes_used,hipMemcpyDeviceToHost,queues_(async)%queueptr))
        else
          call hipCheck(hipMemcpyAsync(record%hostptr,record%deviceptr,&
                  record%num_bytes_used,hipMemcpyDeviceToHost,c_null_ptr))
        endif
      else
#endif
        call hipCheck(hipMemcpy(record%hostptr,record%deviceptr,record%num_bytes_used,hipMemcpyDeviceToHost))
#ifndef BLOCKING_COPIES
      endif
#endif
    end subroutine
    
    function t_record_is_subarray_(record,hostptr,num_bytes,offset_bytes) result(rval)
      use iso_c_binding
      use gpufort_acc_runtime_c_bindings
      implicit none
      class(t_record),intent(inout) :: record
      type(c_ptr), intent(in)       :: hostptr
      integer(c_size_t), intent(in) :: num_bytes
      !
      logical :: rval
      integer(c_size_t),intent(inout),optional :: offset_bytes
      !
      integer(c_size_t) :: opt_offset_bytes
      !
      rval = is_subarray(record%hostptr,record%num_bytes_used, hostptr, num_bytes, opt_offset_bytes)
      if ( present(offset_bytes) ) offset_bytes = opt_offset_bytes
    end function

    !> Release the record for reuse/deallocation.
    subroutine t_record_release_(record)
      use iso_fortran_env
      implicit none
      class(t_record),intent(inout) :: record
      !
      if ( LOG_LEVEL > 1 ) then
        write(output_unit,fmt="(a)",advance="no") "[gpufort-rt][2] release record: "
        flush(output_unit)
        call record%print()
        flush(output_unit)
      endif
      record%hostptr  = c_null_ptr
      record%num_refs = 0
      record%region   = -1
    end subroutine
    
    !> Setup newly created/reused record according to new hostptr.
    subroutine t_record_setup_(record,hostptr,num_bytes,creational_event,region,reuse_existing)
      use iso_c_binding
      use hipfort_check
      use hipfort_hipmalloc
      implicit none
      class(t_record),intent(inout)                       :: record
      type(c_ptr), intent(in)                             :: hostptr
      integer(c_size_t), intent(in)                       :: num_bytes
      integer(kind(gpufort_acc_event_create)), intent(in) :: creational_event
      integer, intent(in)                                 :: region
      logical, intent(in)                                 :: reuse_existing
      !
      record%hostptr           = hostptr
      record%num_refs          = 1
      record%region            = region
      record%creational_event  = creational_event
      record%num_bytes_used    = num_bytes
      if ( .not. reuse_existing ) then
        record_creation_counter_ = record_creation_counter_ + 1
        record%id                = record_creation_counter_ 
        record%num_bytes         = blocked_size(num_bytes)
        call hipCheck(hipMalloc(record%deviceptr,record%num_bytes))
      endif
      if ( creational_event .eq. gpufort_acc_event_copyin .or. &
           creational_event .eq. gpufort_acc_event_copy ) &
             call record%copy_to_device()
    end subroutine
     
    subroutine t_record_destroy_(record)
      use iso_c_binding
      use iso_fortran_env
      use hipfort_check
      use hipfort_hipmalloc
      implicit none
      class(t_record),intent(inout) :: record
      !
      if ( LOG_LEVEL > 1 ) then
        write(output_unit,fmt="(a)",advance="no") "[gpufort-rt][2] destroy record: "
        flush(output_unit)
        call record%print()
        flush(output_unit)
      endif
      call hipCheck(hipFree(record%deviceptr))
      record%deviceptr = c_null_ptr
      record%hostptr   = c_null_ptr
      record%num_refs  = 0
      record%region    = -1
    end subroutine
    
    subroutine t_record_increment_num_refs_(record)
      implicit none
      class(t_record),intent(inout) :: record
      !
      record%num_refs = record%num_refs + 1
    end subroutine
    
    function t_record_decrement_num_refs_(record,threshold) result(ret)
      implicit none
      class(t_record),intent(inout) :: record
      integer,intent(in),optional   :: threshold
      !
      logical :: ret
      !
      record%num_refs = record%num_refs - 1
      ret = record%num_refs <= eval_optval_(threshold,0)
    end function
    
    !
    ! records_
    !
    
    function t_record_list_is_initialized_(record_list) result(ret)
      implicit none
      class(t_record_list),intent(inout) :: record_list
      logical                            :: ret
      !
      ret = allocated(record_list%records)
    end function
    
    subroutine t_record_list_initialize_(record_list)
      implicit none
      class(t_record_list),intent(inout) :: record_list
      !
      type(t_record), allocatable :: new_records(:)
      integer                     :: old_size
      !
      allocate(record_list%records(INITIAL_RECORDS_CAPACITY))
    end subroutine
    
    subroutine t_record_list_grow_(record_list)
      implicit none
      class(t_record_list),intent(inout) :: record_list
      !
      type(t_record), allocatable :: new_records(:)
      integer                     :: old_size
      !
      old_size = size(record_list%records)
      allocate(new_records(old_size*2))
      new_records(1:old_size) = record_list%records(1:old_size)
      deallocate(record_list%records)
      call move_alloc(new_records,record_list%records)
    end subroutine

    subroutine t_record_list_destroy_(record_list)
      implicit none
      class(t_record_list),intent(inout) :: record_list
      !
      integer :: i
      !
      do i = 1, record_list%last_record_index
        if ( record_list%records(i)%is_initialized() ) then
          call record_list%records(i)%destroy()
        endif 
      end do
      deallocate(record_list%records)
    end subroutine

    !> Finds a record for a given host ptr and returns the location.   
    !>
    !> \note Not thread safe
    function t_record_list_find_record_(record_list,hostptr,success) result(loc)
      use iso_fortran_env
      use iso_c_binding
      use gpufort_acc_runtime_c_bindings
      implicit none
      class(t_record_list),intent(inout) :: record_list
      type(c_ptr), intent(in)            :: hostptr
      logical, intent(inout)             :: success
      !
      integer :: i, loc
      !
      if ( .not. c_associated(hostptr) ) ERROR STOP "gpufort_acc_runtime: t_record_list_find_record_: hostptr not c_associated"
      loc     = -1
      success = .false.
      if ( record_list%last_record_index > 0 ) then
        do i = record_list%last_record_index, 1, -1
          if ( record_list%records(i)%is_subarray(hostptr, 0_8) ) then
            loc = i
            success = .true.
            exit ! loop
          endif 
        end do
        if ( LOG_LEVEL > 2 ) then
          write(output_unit,fmt="(a)",advance="no") "[gpufort-rt][3] lookup record: "
          flush(output_unit)
          if ( success ) then
            call record_list%records(loc)%print()
            flush(output_unit)
          else
            write(output_unit,fmt="(a)",advance="yes") "NOT FOUND"
            flush(output_unit)
          endif
        endif
      endif
    end function

    !> Searches first available record from the begin of the record search space.
    !> If it does not find an available record in the current search space,
    !> it takes the record right after the end of the search space
    !> and grows the search space by 1.
    !>
    !> Tries to reuse existing records that have been released.
    !> Checks how many times a record has been considered (unsuccessfully) for reusing and deallocates
    !> associated device data if that has happend NUM_REFS_TO_DEALLOCATE times (or more).
    !>
    !> \note Not thread safe.
    function t_record_list_find_available_record_(record_list,num_bytes,reuse_existing) result(loc)
      implicit none
      class(t_record_list),intent(inout) :: record_list
      integer(c_size_t),intent(in)       :: num_bytes
      logical,intent(inout)              :: reuse_existing
      !
      integer :: i, loc
      !
      reuse_existing = .false.
      loc = record_list%last_record_index+1
      do i = 1, record_list%last_record_index
        if ( .not. record_list%records(i)%is_initialized() ) then ! 1. buffer is empty
          loc = i  
          reuse_existing = .false.
          exit ! exit loop 
        else if ( record_list%records(i)%is_released() ) then ! 2 found a released array
          loc = i  
          if ( record_list%records(i)%num_bytes < num_bytes .or. &
             num_bytes < record_list%records(i)%num_bytes * REUSE_THRESHOLD ) then ! 2.1 buffer too small/big
            ! deallocate routinely unused memory blocks
            if ( record_list%records(i)%decrement_num_refs(NUM_REFS_TO_DEALLOCATE) ) then ! side effect
              record_list%total_memory_bytes = record_list%total_memory_bytes &
                                             - record_list%records(loc)%num_bytes
              call record_list%records(i)%destroy() 
            endif 
            reuse_existing = .false.
          else ! 2. buffer size is fine
            reuse_existing = .true.
          endif
          exit ! exit loop
        endif 
      end do
    end function
    
    !> Inserts a record (inclusive the host-to-device memcpy where required) or increments a record's
    !> reference counter.
    !> 
    !> \note Not thread safe.
    function t_record_list_use_increment_record_(record_list,hostptr,num_bytes,creational_event,async,module_var) result(loc)
      use iso_c_binding
      use iso_fortran_env
      use hipfort
      use hipfort_check
      implicit none
      class(t_record_list),intent(inout)                  :: record_list
      type(c_ptr), intent(in)                             :: hostptr
      integer(c_size_t), intent(in)                       :: num_bytes
      integer(kind(gpufort_acc_event_create)), intent(in) :: creational_event
      integer,intent(in),optional                         :: async
      logical,intent(in),optional                         :: module_var
      !
      integer :: loc 
      logical :: success, reuse_existing
      !
      loc = record_list%find_record(hostptr,success)
      if ( success ) then
         call record_list%records(loc)%increment_num_refs()
      else
         loc = record_list%find_available_record(num_bytes,reuse_existing)
         if (loc .ge. size(record_list%records)) CALL record_list%grow()
         !
         call record_list%records(loc)%setup(hostptr,&
           num_bytes,creational_event,record_list%current_region,reuse_existing)
         if ( eval_optval_(module_var,.false.) ) record_list%records(loc)%region = 0
         if ( creational_event .eq. gpufort_acc_event_copyin .or. &
              creational_event .eq. gpufort_acc_event_copy ) &
                call record_list%records(loc)%copy_to_device(async)
         !print *, loc
         record_list%last_record_index  = max(loc, record_list%last_record_index)
         record_list%total_memory_bytes = record_list%total_memory_bytes + record_list%records(loc)%num_bytes
         if ( LOG_LEVEL > 1 ) then
           if ( reuse_existing ) then
             write(output_unit,fmt="(a)",advance="no") "[gpufort-rt][2] reuse record: "
           else            
             write(output_unit,fmt="(a)",advance="no") "[gpufort-rt][2] create record: "
           endif
           flush(output_unit)
           call record_list%records(loc)%print()
           flush(output_unit)
         endif
      endif
    end function

    !> Deletes a record's reference counter and destroys the record if
    !> the reference counter is zero. Copies the data to the host beforehand
    !> if specified.
    !> 
    !> \note Not thread safe.
    subroutine t_record_list_decrement_release_record_(record_list,hostptr,copy_to_host)
      use iso_c_binding
      use hipfort
      use hipfort_check
      implicit none
      class(t_record_list),intent(inout) :: record_list
      type(c_ptr),intent(in)             :: hostptr
      !
      logical :: copy_to_host
      !
      integer :: loc 
      logical :: success
      !
      loc = record_list%find_record(hostptr,success)
      if ( success ) then
         if ( record_list%records(loc)%decrement_num_refs() ) then
           if ( copy_to_host ) call record_list%records(loc)%copy_to_host() 
         endif
      else
#ifndef DELETE_NORECORD_MEANS_NOOP
        ERROR STOP "gpufort_acc_runtime: t_record_list_decrement_release_record_: could not find matching record for hostptr"
#endif
        return
      endif
      !
    end subroutine
    
    !
    ! public
    !
    !> Ignore the result of a mapping routine.
    !> \param[in] deviceptr a device pointer.
    subroutine gpufort_acc_ignore(deviceptr)
      type(c_ptr),intent(in) :: deviceptr
      ! 
      ! nop  
    end subroutine

    subroutine gpufort_acc_init()
      implicit none
      integer :: j
      character(len=255) :: tmp
      !
      if ( initialized_ ) then 
        ERROR STOP "gpufort_acc_init: runtime already initialized"
      else
        call get_environment_variable("GPUFORT_LOG_LEVEL", tmp)
        if (len_trim(tmp) > 0) read(tmp,*) LOG_LEVEL
        !
        call get_environment_variable("GPUFORT_MAX_QUEUES", tmp)
        if (len_trim(tmp) > 0) read(tmp,*) MAX_QUEUES
        call get_environment_variable("GPUFORT_INITIAL_RECORDS_CAPACITY", tmp)
        if (len_trim(tmp) > 0) read(tmp,*) INITIAL_RECORDS_CAPACITY
        !
        call get_environment_variable("GPUFORT_BLOCK_SIZE", tmp)
        if (len_trim(tmp) > 0) read(tmp,*) BLOCK_SIZE
        call get_environment_variable("GPUFORT_REUSE_THRESHOLD", tmp)
        if (len_trim(tmp) > 0) read(tmp,*) REUSE_THRESHOLD
        call get_environment_variable("GPUFORT_NUM_REFS_TO_DEALLOCATE", tmp)
        if (len_trim(tmp) > 0) read(tmp,*) NUM_REFS_TO_DEALLOCATE
        if ( LOG_LEVEL > 0 ) then 
          write(*,*) "GPUFORT_LOG_LEVEL=",LOG_LEVEL 
          write(*,*) "GPUFORT_MAX_QUEUES=",MAX_QUEUES
          write(*,*) "GPUFORT_INITIAL_RECORDS_CAPACITY=",INITIAL_RECORDS_CAPACITY
          write(*,*) "GPUFORT_BLOCK_SIZE=",BLOCK_SIZE 
          write(*,*) "GPUFORT_REUSE_THRESHOLD=",REUSE_THRESHOLD 
          write(*,*) "GPUFORT_NUM_REFS_TO_DEALLOCATE=",NUM_REFS_TO_DEALLOCATE 
        endif 
        !
        call record_list_%initialize()
        allocate(queues_(1:MAX_QUEUES))
        initialized_ = .true.
      endif
    end subroutine
   
    !> Deallocate all host data and the records_ and
    !> queue data structures.. 
    subroutine gpufort_acc_shutdown()
      implicit none
      integer :: i, j
      if ( .not. initialized_ ) ERROR STOP "gpufort_acc_shutdown: runtime not initialized"
      ! deallocate records_
      call record_list_%destroy()
      ! deallocate queues_ elements
      do i = 1, last_queue_index_
        if ( queues_(i)%is_initialized() ) then
          call queues_(i)%destroy()
        endif 
      end do
    end subroutine

    !> Summary The wait directive causes the local thread to wait for completion of asynchronous
    !> operations on the current device, such as an accelerator parallel, kernels, or serial region or an
    !> update directive, or causes one device activity queue to synchronize with 
    !> one or more other activity queues on the current device.
    !> 
    !> Syntax 
    !> 
    !> In Fortran the syntax of the wait directive is:
    !> 
    !> !$acc wait [( int-expr-list )] [clause-list]
    !> 
    !> where clause is:
    !> 
    !> async [( int-expr )]
    !> 
    !> The wait argument, if it appears, must be one or more async-arguments.
    !> If there is no wait argument and no async clause, the local thread will wait until all operations
    !> enqueued by this thread on any activity queue on the current device have completed.
    !> If there are one or more int-expr expressions and no async clause, the local thread will wait until all
    !> operations enqueued by this thread on each of the associated device activity queues have completed.
    !> If there are two or more threads executing and sharing the same device, a wait directive with no
    !> async clause will cause the local thread to wait until all of the appropriate asynchronous operations previously enqueued by that thread have completed. 
    !> To guarantee that operations have been
    !> enqueued by other threads requires additional synchronization between those threads. There is no
    !> guarantee that all the similar asynchronous operations initiated by other threads will have completed.
    !> 
    !> If there is an async clause, no new operation may be launched or executed on the async activity queue on
    !> the current device until all operations enqueued up to this point by this thread on the
    !> asynchronous activity queues associated with the wait argument have completed. One legal implementation is for the 
    !> local thread to wait for all the associated asynchronous device activity queues.
    !> Another legal implementation is for the thread to enqueue a synchronization operation in such a
    !> way that no new operation will start until the operations enqueued on the associated asynchronous
    !> device activity queues have completed.
    !> A wait directive is functionally equivalent to a call to one of the acc_wait, acc_wait_async,
    !> acc_wait_all or acc_wait_all_async runtime API routines, as described in Sections 3.2.11,
    !> 3.2.12, 3.2.13 and 3.2.14.
    subroutine gpufort_acc_wait(arg,async)
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
      if ( .not. initialized_ ) ERROR STOP "gpufort_acc_wait: runtime not initialized"
      no_opt_present = .true.
      if ( present(arg) ) then
        no_opt_present = .false.
        do i=1,size(arg)
            call hipCheck(hipStreamSynchronize(queues_(arg(i))%queueptr))
        end do
      endif
      !
      if ( present(async) ) then
        no_opt_present = .false.
        do i=1,size(async)
          ! acc_wait_async This function enqueues a wait operation on the queue async for any and all asynchronous operations that have been previously enqueued on any queue.
          !queues_(async(i))%enqueue_wait_event()
          call hipCheck(hipStreamSynchronize(queues_(async(i))%queueptr))
        end do
      endif
      !
      if ( no_opt_present ) then 
        do i=1,last_queue_index_
          if ( queues_(i)%is_initialized() ) then
            call hipCheck(hipStreamSynchronize(queues_(i)%queueptr))
          endif
        end do
        call hipCheck(hipStreamSynchronize(c_null_ptr)) ! synchronize default stream
      endif
    end subroutine
    
    !> \note We can use this also for a "normal" data section 
    !> \note We can use this also for any other region that uses device data
    subroutine gpufort_acc_enter_region(unstructured,implicit_region)
      implicit none
      logical,intent(in),optional :: unstructured
      logical,intent(in),optional :: implicit_region
      if ( .not. initialized_ ) call gpufort_acc_init()
      record_list_%current_region = record_list_%current_region + 1
    end subroutine
 
    !> \note We can use this also for a "normal" end data section 
    !> \note We can use this also for any other region exit that uses device data
    subroutine gpufort_acc_exit_region(unstructured,implicit_region)
#ifdef EXIT_REGION_SYNCHRONIZE_DEVICE
      use hipfort
      use hipfort_check
#endif
      implicit none
      logical,intent(in),optional :: unstructured
      logical,intent(in),optional :: implicit_region
      !
      integer :: i, new_last_record_index
      !
#ifdef EXIT_REGION_SYNCHRONIZE_DEVICE
     call hipCheck(hipDeviceSynchronize())
#endif
#ifdef EXIT_REGION_SYNCHRONIZE_STREAMS
     call gpufort_acc_wait()
#endif
      new_last_record_index = 0
      do i = 1, record_list_%last_record_index
        if ( record_list_%records(i)%is_initialized() ) then
          if ( record_list_%records(i)%region .eq. record_list_%current_region ) then
            if ( record_list_%records(i)%creational_event .eq. gpufort_acc_event_create .or.&
                 record_list_%records(i)%creational_event .eq. gpufort_acc_event_copyin ) then
              call record_list_%records(i)%release()
            else if ( record_list_%records(i)%creational_event .eq. gpufort_acc_event_copyout .or. &
                 record_list_%records(i)%creational_event .eq. gpufort_acc_event_copy ) then
              call record_list_%records(i)%copy_to_host() ! synchronous
              call record_list_%records(i)%release()
            endif
          endif
          new_last_record_index = i
        endif 
      end do
      record_list_%last_record_index = new_last_record_index
      !
      record_list_%current_region = record_list_%current_region -1
      !
    end subroutine

    ! Data Clauses
    ! The description applies to the clauses used on compute
    ! constructs, data constructs, and enter data and exit
    ! data directives. Data clauses may not follow a device_type
    ! clause. These clauses have no effect on a shared memory device.
    
    !> present( list ) parallel, kernels, serial, data, declare
    !> When entering the region, the data must be present in device
    !> memory, and the structured reference count is incremented.
    !> When exiting the region, the structured reference count is
    !> decremented.
    !>
    !> \note We just return the device pointer here and do not modify the counters.
    function gpufort_acc_present_b(hostptr,num_bytes,module_var,or,async) result(deviceptr)
      use iso_fortran_env
      use iso_c_binding
      use gpufort_acc_runtime_c_bindings
      implicit none
      type(c_ptr),intent(in)       :: hostptr
      integer(c_size_t),intent(in) :: num_bytes
      !logical,intent(in),optional :: exiting
      logical,intent(in),optional  :: module_var
      integer(kind(gpufort_acc_event_undefined)),&
               intent(in),optional :: or
      integer,intent(in),optional  :: async
      !
      type(c_ptr) :: deviceptr
      !
      integer(c_size_t) :: offset_bytes
      logical           :: success, fits
      integer(kind(gpufort_acc_event_undefined)) &
                        :: opt_or
      integer           :: loc, num_lists_to_check
      !
      if ( .not. initialized_ .and. eval_optval_(module_var,.false.) ) call gpufort_acc_init()
      if ( .not. initialized_ ) ERROR STOP "gpufort_acc_present_b: runtime not initialized"
      if ( .not. c_associated(hostptr) ) then
        ERROR STOP "gpufort_acc_present_b: hostptr not c_associated"
      else
        loc = record_list_%find_record(hostptr,success) ! TODO already detect a suitable candidate here on-the-fly
        if ( success ) then 
          fits      = record_list_%records(loc)%is_subarray(hostptr,num_bytes,offset_bytes)
          deviceptr = inc_cptr(record_list_%records(loc)%deviceptr,offset_bytes)
        else
          offset_bytes = -1
          success      = .true.
          opt_or = eval_optval_(or,gpufort_acc_event_undefined)
          select case (opt_or)
            case (gpufort_acc_event_create)
              deviceptr = gpufort_acc_create_b(hostptr,num_bytes,async,module_var)
            case (gpufort_acc_event_copyin)
              deviceptr = gpufort_acc_copyin_b(hostptr,num_bytes,async,module_var)
            case (gpufort_acc_event_copyout)
              deviceptr = gpufort_acc_copyout_b(hostptr,num_bytes,async,module_var)
            case (gpufort_acc_event_copy)
              deviceptr = gpufort_acc_copy_b(hostptr,num_bytes,async,module_var)
            case default
              success = .false.
              print *, "ERROR: did not find record for hostptr:"
              CALL print_cptr(hostptr)
              ERROR STOP "gpufort_acc_present_b: no record found for hostptr"
          end select
        endif
        if ( LOG_LEVEL > 0 .and. success ) then
          write(output_unit,fmt="(a)",advance="no") "[gpufort-rt][1] gpufort_acc_present_b: retrieved deviceptr="
          flush(output_unit)
          CALL print_cptr(deviceptr)
          flush(output_unit)
          write(output_unit,fmt="(a,i0.0,a)",advance="no") "(offset_bytes=",offset_bytes,",base deviceptr="
          flush(output_unit)
          CALL print_cptr(record_list_%records(loc)%deviceptr)
          flush(output_unit)
          write(output_unit,fmt="(a,i0.0)",advance="no") ",base num_bytes=", record_list_%records(loc)%num_bytes
          write(output_unit,fmt="(a)",advance="no") ") for hostptr="
          flush(output_unit)
          CALL print_cptr(hostptr)
          flush(output_unit)
          print *, ""
        endif
        !if (exiting) then 
        !  record_list_%records(loc)%decrement_num_refs()
        !else
        !  record_list_%records(loc)%increment_num_refs()
        !endif
      endif
    end function
    
    function gpufort_acc_present_or_create_b(hostptr,num_bytes,module_var,async) result(deviceptr)
      use iso_fortran_env
      use iso_c_binding
      use gpufort_acc_runtime_c_bindings
      implicit none
      type(c_ptr),intent(in)       :: hostptr
      integer(c_size_t),intent(in) :: num_bytes
      !logical,intent(in),optional :: exiting
      logical,intent(in),optional  :: module_var
      integer,intent(in),optional  :: async
      !
      deviceptr = gpufort_acc_present_b(hostptr,num_bytes,module_var,gpufort_acc_event_create,async) 
    end function
    
    function gpufort_acc_present_or_copyin_b(hostptr,num_bytes,module_var,async) result(deviceptr)
      use iso_fortran_env
      use iso_c_binding
      use gpufort_acc_runtime_c_bindings
      implicit none
      type(c_ptr),intent(in)       :: hostptr
      integer(c_size_t),intent(in) :: num_bytes
      !logical,intent(in),optional :: exiting
      logical,intent(in),optional  :: module_var
      integer,intent(in),optional  :: async
      !
      deviceptr = gpufort_acc_present_b(hostptr,num_bytes,module_var,gpufort_acc_event_copyin,async) 
    end function
    
    function gpufort_acc_present_or_copyout_b(hostptr,num_bytes,module_var,async) result(deviceptr)
      use iso_fortran_env
      use iso_c_binding
      use gpufort_acc_runtime_c_bindings
      implicit none
      type(c_ptr),intent(in)       :: hostptr
      integer(c_size_t),intent(in) :: num_bytes
      !logical,intent(in),optional :: exiting
      logical,intent(in),optional  :: module_var
      integer,intent(in),optional  :: async
      !
      deviceptr = gpufort_acc_present_b(hostptr,num_bytes,module_var,gpufort_acc_event_copyout,async) 
    end function
    
    function gpufort_acc_present_or_copy_b(hostptr,num_bytes,module_var,async) result(deviceptr)
      use iso_fortran_env
      use iso_c_binding
      use gpufort_acc_runtime_c_bindings
      implicit none
      type(c_ptr),intent(in)       :: hostptr
      integer(c_size_t),intent(in) :: num_bytes
      !logical,intent(in),optional :: exiting
      logical,intent(in),optional  :: module_var
      integer,intent(in),optional  :: async
      !
      deviceptr = gpufort_acc_present_b(hostptr,num_bytes,module_var,gpufort_acc_event_copy,async) 
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
    function gpufort_acc_create_b(hostptr,num_bytes,async,module_var) result(deviceptr)
      use iso_c_binding
      implicit none
      type(c_ptr),intent(in)       :: hostptr
      integer(c_size_t),intent(in) :: num_bytes
      integer,intent(in),optional  :: async ! ignored for now
      logical,intent(in),optional  :: module_var
      type(c_ptr) :: deviceptr
      !
      integer :: loc
      !
      if ( .not. initialized_ .and. eval_optval_(module_var,.false.) ) call gpufort_acc_init()
      if ( .not. initialized_ ) ERROR STOP "gpufort_acc_create_b: runtime not initalized"
      if ( .not. c_associated(hostptr) ) then
        ERROR STOP "gpufort_acc_create_b: hostptr not c_associated"
      else
        loc       = record_list_%use_increment_record(hostptr,num_bytes,&
                      gpufort_acc_event_create,async,module_var)
        deviceptr = record_list_%records(loc)%deviceptr
      endif
    end function
    
    !> no_create( list ) parallel, kernels, serial, data
    !> When entering the region, if the data in list is already present on
    !> the current device, the structured reference count is incremented
    !> and that copy is used. Otherwise, no action is performed and any
    !> device code in the construct will use the local memory address
    !> for that data.
    !>
    !> \note We just return the device pointer here (or c_null_ptr) and do not modify the counters.
    function gpufort_acc_no_create_b(hostptr,module_var) result(deviceptr)
      use iso_c_binding
      implicit none
      type(c_ptr), intent(in)     :: hostptr
      logical,intent(in),optional :: module_var
      type(c_ptr) :: deviceptr
      !
      logical :: success
      integer :: loc
      !
      if ( .not. initialized_ ) ERROR STOP "gpufort_acc_create_b: runtime not initalized"
      if ( .not. c_associated(hostptr) ) then
        ERROR STOP "gpufort_acc_no_create_b: hostptr not c_associated"
      else
        loc = record_list_%find_record(hostptr,success)
        deviceptr = c_null_ptr
        if ( success ) deviceptr = record_list_%records(loc)%deviceptr
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
    subroutine gpufort_acc_delete_b(hostptr,finalize,module_var)
      use iso_c_binding
      implicit none
      type(c_ptr), intent(in)     :: hostptr
      logical,intent(in),optional :: finalize, module_var
      !
      logical :: success, opt_finalize 
      integer :: loc
      !
      if ( .not. initialized_ ) ERROR STOP "gpufort_acc_delete_b: runtime not initialized"
      if ( .not. c_associated(hostptr) ) then
#ifndef NULLPTR_MEANS_NOOP
        ERROR STOP "gpufort_acc_delete_b: hostptr not c_associated"
#endif
        return
      else
        opt_finalize = .false.
        if ( present(finalize) ) opt_finalize = finalize
        if ( opt_finalize ) then
          loc = record_list_%find_record(hostptr,success)
          if ( .not. success ) ERROR STOP "gpufort_acc_delete: no record found for hostptr"
          call record_list_%records(loc)%release()
        else
          call record_list_%decrement_release_record(hostptr,copy_to_host=.false.)
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
    !> \note Exiting case handled in gpufort_acc_exit_region
    function gpufort_acc_copyin_b(hostptr,num_bytes,async,module_var) result(deviceptr)
      use iso_c_binding
      implicit none
      type(c_ptr), intent(in)      :: hostptr
      integer(c_size_t),intent(in) :: num_bytes
      integer,intent(in),optional  :: async
      logical,intent(in),optional  :: module_var
      !
      type(c_ptr) :: deviceptr 
      !
      integer :: loc
      !
      if ( .not. initialized_ .and. eval_optval_(module_var,.false.) ) call gpufort_acc_init()
      if ( .not. initialized_ ) ERROR STOP "gpufort_acc_copyin_b: runtime not initialized"
      if ( .not. c_associated(hostptr) ) then
#ifndef NULLPTR_MEANS_NOOP
        ERROR STOP "gpufort_acc_copyin_b: hostptr not c_associated"
#endif
        deviceptr = c_null_ptr
      else
        loc       = record_list_%use_increment_record(hostptr,num_bytes,gpufort_acc_event_copyin,async,module_var)
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
    !> \note Exiting case handled in gpufort_acc_exit_region
    function gpufort_acc_copyout_b(hostptr,num_bytes,async,module_var) result(deviceptr)
      use iso_c_binding
      implicit none
      ! Return type(c_ptr)
      type(c_ptr), intent(in)      :: hostptr
      integer(c_size_t),intent(in) :: num_bytes
      integer,intent(in),optional  :: async
      logical,intent(in),optional  :: module_var
      !
      type(c_ptr) :: deviceptr 
      !
      integer :: loc
      !
      if ( .not. initialized_ .and. eval_optval_(module_var,.false.) ) call gpufort_acc_init()
      if ( .not. initialized_ ) ERROR STOP "gpufort_acc_copyout_b: runtime not initialized"
      if ( .not. c_associated(hostptr) ) then
#ifndef NULLPTR_MEANS_NOOP
        ERROR STOP "gpufort_acc_copyout_b: hostptr not c_associated"
#endif
        deviceptr = c_null_ptr
      else
        loc       = record_list_%use_increment_record(hostptr,num_bytes,gpufort_acc_event_copyout,async,module_var)
        deviceptr = record_list_%records(loc)%deviceptr
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
    !> \note Exiting case handled in gpufort_acc_exit_region
    function gpufort_acc_copy_b(hostptr,num_bytes,async,module_var) result(deviceptr)
      use iso_c_binding
      implicit none
      ! Return type(c_ptr)
      type(c_ptr), intent(in)      :: hostptr
      integer(c_size_t),intent(in) :: num_bytes
      integer,intent(in),optional  :: async
      logical,intent(in),optional  :: module_var
      !
      type(c_ptr) :: deviceptr 
      !
      integer :: loc
      !
      if ( .not. initialized_ .and. eval_optval_(module_var,.false.) ) call gpufort_acc_init()
      if ( .not. initialized_ ) ERROR STOP "gpufort_acc_copy_b: runtime not initialized"
      if ( .not. c_associated(hostptr) ) then
#ifndef NULLPTR_MEANS_NOOP
        ERROR STOP "gpufort_acc_copy_b: hostptr not c_associated"
#endif
        deviceptr = c_null_ptr
      else
        loc       = record_list_%use_increment_record(hostptr,num_bytes,&
                     gpufort_acc_event_copy,async,module_var)
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
    subroutine gpufort_acc_update_host_b(hostptr,condition,if_present,async,module_var)
      use iso_c_binding
      implicit none
      type(c_ptr),intent(in)      :: hostptr
      logical,intent(in),optional :: condition, if_present, module_var
      integer,intent(in),optional :: async
      !
      integer :: loc
      !
      logical :: success, opt_condition, opt_if_present 
      !
      if ( .not. initialized_ ) ERROR STOP "gpufort_acc_update_host_b: runtime not initialized"
      if ( .not. c_associated(hostptr) ) then
#ifndef NULLPTR_MEANS_NOOP
        ERROR STOP "gpufort_acc_update_host_b: hostptr not c_associated"
#endif
        return
      endif
      opt_condition  = .true.
      opt_if_present = .false.
      if ( present(condition) )  opt_condition  = condition
      if ( present(if_present) ) opt_if_present = if_present
      !
      if ( opt_condition ) then
        loc = record_list_%find_record(hostptr,success)
        !
        if ( .not. success .and. .not. opt_if_present ) ERROR STOP "gpufort_acc_update_host_b: no deviceptr found for hostptr"
        ! 
        if ( success .and. present(async) ) then 
          call record_list_%records(loc)%copy_to_host(async)
        else if ( success ) then
          call record_list_%records(loc)%copy_to_host()
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
    subroutine gpufort_acc_update_device_b(hostptr,condition,if_present,async,module_var)
      use iso_c_binding
      implicit none
      type(c_ptr),intent(in)      :: hostptr
      logical,intent(in),optional :: condition, if_present, module_var
      integer,intent(in),optional :: async
      !
      integer :: loc
      logical :: success, opt_condition, opt_if_present
      !
      if ( .not. initialized_ ) ERROR STOP "gpufort_acc_update_device_b: runtime not initialized"
      if ( .not. c_associated(hostptr) ) then
#ifndef NULLPTR_MEANS_NOOP
        ERROR STOP "gpufort_acc_update_device_b: deviceptr not c_associated"
#endif
        return
      endif
      opt_condition  = .true.
      opt_if_present = .false.
      if ( present(condition) )  opt_condition  = condition
      if ( present(if_present) ) opt_if_present = if_present
      !
      if ( opt_condition ) then
        loc = record_list_%find_record(hostptr,success)
        !
        if ( .not. success .and. .not. opt_if_present ) ERROR STOP "gpufort_acc_update_device_b: no deviceptr found for hostptr"
        ! 
        if ( success .and. present(async) ) then 
          call record_list_%records(loc)%copy_to_device(async)
        else if ( success ) then
          call record_list_%records(loc)%copy_to_device()
        endif
      endif
    end subroutine

    function gpufort_acc_get_stream(queue_id):
       implicit none
       integer, intent(in) :: queue_id
       if ( queue_id .eq. 0 ) then
          return c_null_ptr
       else 
          call ensure_queue_exists_(queue_id)
          return queue_id(id)%queueptr
       endif
    end function
end module
