! SPDX-License-Identifier: MIT                                                
! Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
! TODO not thread-safe, make use of OMP_LIB module if this becomes a problem

! TODO will be slow for large pointer storages 
! If this becomes an issue, use faster data structures or bind the public interfaces to
! a faster C runtime

#define blocked_size(num_bytes) ((((num_bytes)+BLOCK_SIZE-1)/BLOCK_SIZE) * BLOCK_SIZE)

module gpufort_acc_runtime_base
  use iso_c_binding

  ! These routines are public

  public :: gpufort_acc_copyin_b, gpufort_acc_copyout_b, gpufort_acc_copy_b, & 
            gpufort_acc_create_b, gpufort_acc_no_create_b, gpufort_acc_present_b, &
            gpufort_acc_delete_b 
  public :: gpufort_acc_update_host_b, gpufort_acc_update_device_b
  public :: gpufort_acc_init, gpufort_acc_shutdown
  public :: gpufort_acc_enter_region, gpufort_acc_exit_region
  public :: gpufort_acc_wait
  public :: gpufort_acc_runtime_print_summary
  
  public :: gpufort_acc_runtime_record_exists, gpufort_acc_runtime_get_record_id

  PRIVATE ! Everything else is private

  ! 
  ! parameters
  !

  integer, parameter :: MAX_QUEUES = 64
  integer, parameter :: INITIAL_RECORDS_CAPACITY = 4096
  integer, parameter :: BLOCK_SIZE = 32

  !
  ! members
  !
  
  integer :: record_creation_counter_        = 1

  logical, save :: initialized_              = .FALSE.

  integer, save :: last_record_index_        = 0
  integer, save :: last_queue_index_         = 0
  
  integer, save :: current_region_           = 0
  integer(c_size_t), save :: total_memory_b_ = 0
  
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
    integer :: id                  = 1
    type(c_ptr) :: hostptr         = c_null_ptr
    type(c_ptr) :: deviceptr       = c_null_ptr
    integer(c_size_t) :: num_bytes = 0
    integer :: num_refs            = 0
    integer :: region              = 0
    integer(kind(gpufort_acc_event_create)) :: creational_event = gpufort_acc_event_undefined

    contains 
      procedure :: print => t_record_print_
      procedure :: is_initialized => t_record_is_initialized_
      procedure :: is_subarray => t_record_is_subarray_
      procedure :: initialize => t_record_initialize_
      procedure :: destroy => t_record_destroy_
      procedure :: copy_to_device => t_record_copy_to_device_
      procedure :: copy_to_host   => t_record_copy_to_host_
      procedure :: decrement_num_refs => t_record_decrement_num_refs_
      procedure :: increment_num_refs => t_record_increment_num_refs_
  end type

  !> Data structure to map integer numbers to queues.  
  type :: t_queue
    type(c_ptr) :: queueptr = c_null_ptr
    integer     :: num_refs  = 0
    
    contains 
      procedure :: is_initialized => t_queue_is_initialized_
      procedure :: initialize => t_queue_initialize_
      procedure :: destroy => t_queue_destroy_
      procedure :: decrement_num_refs => t_queue_decrement_num_refs_
      procedure :: increment_num_refs => t_queue_increment_num_refs_
  end type

  type(t_record), save, allocatable, private :: records_(:)
  type(t_queue), save, private :: queues_(MAX_QUEUES)

  contains
    
    !
    ! debugging/analysis
    !
  
    function gpufort_acc_runtime_record_exists(hostptr,id,print_record) result(success)
      use iso_c_binding 
      implicit none
      type(c_ptr),intent(in),optional :: hostptr
      integer,intent(in),optional     :: id
      logical,intent(in),optional     :: print_record
      !
      logical :: success
      !
      integer :: i
      logical :: opt_print_record
      !
      opt_print_record = .FALSE.
      !
      if ( present(print_record) ) opt_print_record = print_record
      !
      success = .FALSE.
      do i = 1, last_record_index_
        if ( ( present(hostptr) .and. records_(i)%is_subarray(hostptr,0_8) ) .or. &
             ( present(id) .and. records_(i)%id .eq. id ) ) then
          success = .TRUE.
          if ( opt_print_record ) CALL records_(i)%print() 
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
      do i = 1, last_record_index_
        if ( records_(i)%is_subarray(hostptr,0_8) ) then
          id = records_(i)%id
          exit ! loop
        endif 
      end do
    end function
   
    subroutine gpufort_acc_runtime_print_summary(print_records)
      use iso_c_binding
      use hipfort
      use hipfort_check
      implicit none
      logical,optional,intent(in) :: print_records
      !
      integer :: i
      integer(c_size_t) :: free_memory, total_memory 
      !
      CALL hipCheck(hipMemGetInfo(free_memory,total_memory))
      !
      print *, "SUMMARY"
      print *, ""
      print *, "globals:"
      print *, "- current region             = ", current_region_
      print *, "- last record index          = ", last_record_index_
      print *, "- last queue index           = ", last_queue_index_
      print *, "- total records created      = ", record_creation_counter_
      print *, "- total memory allocated (B) = ", total_memory_b_
      print *, "- HIP used memory        (B) = ", (total_memory - free_memory), "/", total_memory
      print *, ""
      if ( present(print_records) .and. print_records ) then
        print *, "records:"
        do i = 1, last_record_index_
           print *, "- record ", i, ":"
           CALL records_(i)%print()
        end do
      end if
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
      queue%num_refs = 1
    end subroutine
    
    subroutine t_queue_destroy_(queue)
      use hipfort
      use hipfort_check
      implicit none
      class(t_queue),intent(inout) :: queue
      !
      call hipCheck(hipStreamDestroy(queue%queueptr))
      queue%queueptr = c_null_ptr
      queue%num_refs = 0
    end subroutine
    
    subroutine t_queue_increment_num_refs_(queue)
      implicit none
      class(t_queue),intent(inout) :: queue
      !
      queue%num_refs = queue%num_refs + 1
    end subroutine
    
    function t_queue_decrement_num_refs_(queue) result(ret)
      implicit none
      class(t_queue),intent(inout) :: queue
      logical :: ret
      !
      queue%num_refs = queue%num_refs - 1
      ret = queue%num_refs .eq. 0
    end function
    
    !
    ! queues_
    !
    
    !> \note Not thread safe 
    subroutine create_increment_queue_(id)
       implicit none
       integer, intent(in) :: id
       !
       if ( id .gt. 0 .and. id .le. MAX_QUEUES ) then
          if ( .not. queues_(id)%is_initialized() ) then
            call queues_(id)%initialize()
          else 
            call queues_(id)%increment_num_refs()
          endif
          last_queue_index_ = max(id, last_queue_index_)
       else if ( id .gt. MAX_QUEUES ) then
         ERROR STOP "gpufort_acc_runtime: create_increment_queue_: queue id greater than parameter MAX_QUEUES" 
       endif
    end subroutine
   
    !> \note Not thread safe 
    subroutine decrement_delete_queue_(id)
       implicit none
       integer, intent(in) :: id
       !
       if ( id .gt. 0 .and. id .le. MAX_QUEUES ) then
          if ( queues_(id)%is_initialized() ) then
            if ( queues_(id)%decrement_num_refs() ) then ! side effects
              call queues_(id)%destroy()
            endif
          endif
          if ( id .eq. last_queue_index_ ) then
            last_queue_index_ = last_queue_index_ - 1
          endif
       else if ( id .gt. MAX_QUEUES ) then
         ERROR STOP "gpufort_acc_runtime: decrement_delete_queue_: queue id greater than parameter MAX_QUEUES"
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
      ret = c_associated( record%hostptr )
    end function
    
    subroutine t_record_copy_to_device_(record,async)
      use iso_c_binding
      use hipfort_check
      use hipfort_hipmemcpy
      implicit none
      class(t_record),intent(in)  :: record
      integer,optional,intent(in) :: async
      !
#ifndef BLOCKING_COPIES
      if ( present(async) ) then
        if ( async .ge. 0 ) then
          call create_increment_queue_(async)
          call hipCheck(hipMemcpyAsync(record%deviceptr,record%hostptr,&
                  record%num_bytes,hipMemcpyHostToDevice,queues_(async)%queueptr))
        else
          call hipCheck(hipMemcpyAsync(record%deviceptr,record%hostptr,&
                  record%num_bytes,hipMemcpyHostToDevice,c_null_ptr))
        endif
      else
#endif
        call hipCheck(hipMemcpy(record%deviceptr,record%hostptr,record%num_bytes,hipMemcpyHostToDevice))
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
      integer,optional,intent(in) :: async
      !
#ifndef BLOCKING_COPIES
      if ( present(async) ) then
        if ( async .ge. 0 ) then
          call create_increment_queue_(async)
          call hipCheck(hipMemcpyAsync(record%hostptr,record%deviceptr,&
                  record%num_bytes,hipMemcpyDeviceToHost,queues_(async)%queueptr))
        else
          call hipCheck(hipMemcpyAsync(record%hostptr,record%deviceptr,&
                  record%num_bytes,hipMemcpyDeviceToHost,c_null_ptr))
        endif
      else
#endif
        call hipCheck(hipMemcpy(record%hostptr,record%deviceptr,record%num_bytes,hipMemcpyDeviceToHost))
#ifndef BLOCKING_COPIES
      endif
#endif
    end subroutine
    
    function t_record_is_subarray_(record,hostptr,num_bytes) result(rval)
      use iso_c_binding
      use gpufort_acc_runtime_c_bindings
      implicit none
      class(t_record),intent(inout) :: record
      type(c_ptr), intent(in)       :: hostptr
      integer(c_size_t), intent(in) :: num_bytes
      !
      logical :: rval
      integer(c_size_t) :: offset_bytes
      !
      rval = is_subarray(record%hostptr,record%num_bytes, hostptr, num_bytes, offset_bytes)
    end function
    
    subroutine t_record_initialize_(record,hostptr,num_bytes,creational_event,current_region)
      use iso_c_binding
      use hipfort_check
      use hipfort_hipmalloc
      implicit none
      class(t_record),intent(inout)                       :: record
      type(c_ptr), intent(in)                             :: hostptr
      integer(c_size_t), intent(in)                       :: num_bytes
      integer(kind(gpufort_acc_event_create)), intent(in) :: creational_event
      integer, intent(in)                                 :: current_region
      !
      record%hostptr          = hostptr
      record%num_refs         = 1
      record%num_bytes        = num_bytes
      record%region           = current_region
      record%creational_event = creational_event
      call hipCheck(hipMalloc(record%deviceptr,blocked_size(num_bytes)))
      if ( creational_event .eq. gpufort_acc_event_copyin .or. &
           creational_event .eq. gpufort_acc_event_copy ) &
             call record%copy_to_device()

      record%id = record_creation_counter_ 
      record_creation_counter_ = record_creation_counter_ + 1
    end subroutine
     
    subroutine t_record_destroy_(record)
      use iso_c_binding
      use hipfort_check
      use hipfort_hipmalloc
      implicit none
      class(t_record),intent(inout) :: record
      !
      call hipCheck(hipFree(record%deviceptr))
      record%deviceptr = c_null_ptr
      record%hostptr = c_null_ptr
      record%num_refs = 0
      record%region = -1
    end subroutine
    
    subroutine t_record_increment_num_refs_(record)
      implicit none
      class(t_record),intent(inout) :: record
      !
      record%num_refs = record%num_refs + 1
    end subroutine
    
    function t_record_decrement_num_refs_(record) result(ret)
      implicit none
      class(t_record),intent(inout) :: record
      !
      logical :: ret
      !
      record%num_refs = record%num_refs - 1
      ret = record%num_refs .eq. 0
    end function
    
    !
    ! records_
    !
    
    subroutine grow_records_()
      implicit none
      integer :: old_size
      type(t_record), save, allocatable :: new_records(:)
      !
      old_size = SIZE(records_)
      allocate(new_records(old_size*2))
      new_records(1:old_size) = records_(1:old_size)
      deallocate(records_)
      call move_alloc(new_records,records_)
    end subroutine

    !> Finds a record for a given host ptr and returns the location.   
    !>
    !> \note Not thread safe
    function find_record_(hostptr,success) result(loc)
      use iso_fortran_env
      use iso_c_binding
      use gpufort_acc_runtime_c_bindings
      implicit none
      type(c_ptr), intent(in) :: hostptr
      logical, intent(inout) :: success
      !
      integer :: i, loc
      !
      if ( .not. c_associated(hostptr) ) ERROR STOP "gpufort_acc_runtime: find_record_: hostptr not c_associated"
      loc = -1
      success = .FALSE.
      do i = 1, last_record_index_
        !if ( c_associated(records_(i)%hostptr, hostptr) ) then
        if ( records_(i)%is_subarray(hostptr, 0_8) ) then
          loc = i
          success = .TRUE.
          exit ! loop
        endif 
      end do
#if DEBUG > 2
      write(output_unit,fmt="(a)",advance="no") "[3]lookup record: "
      flush(output_unit)
      if ( success ) then
        call records_(loc)%print()
      else
        write(output_unit,fmt="(a)",advance="yes") "NOT FOUND"
      endif
#endif
    end function

    !> Searches first empty record from the begin of the record search space.
    !> If it does not find an empty record in the current search space,
    !> it takes the record right after the end of the search space
    !> and grows the search space by 1.
    !>
    !> \note Not thread safe.
    function find_first_empty_record_() result(loc)
      implicit none
      integer :: i, loc
      !
      do i = 1, last_record_index_ + 1
        if ( .not. records_(i)%is_initialized() ) then
          loc = i
          exit ! exit loop
        endif 
      end do
    end function
    
    !> Inserts a record (inclusive the host-to-device memcpy where required) or increments a record's
    !> reference counter.
    !> 
    !> \note Not thread safe.
    function insert_increment_record_(hostptr,num_bytes,creational_event,async) result(loc)
      use iso_c_binding
      use iso_fortran_env
      use hipfort
      use hipfort_check
      implicit none
      type(c_ptr), intent(in)                             :: hostptr
      integer(c_size_t), intent(in)                       :: num_bytes
      integer(kind(gpufort_acc_event_create)), intent(in) :: creational_event
      integer,optional,intent(in)                         :: async
      !
      integer :: loc 
      logical :: success
      !
      loc = find_record_(hostptr,success)
      if ( success ) then
         call records_(loc)%increment_num_refs()
      else
         loc = find_first_empty_record_()
         if (loc .ge. size(records_)) CALL grow_records_()
         !
         call records_(loc)%initialize(hostptr,&
                 num_bytes,creational_event,current_region_)
         if ( creational_event .eq. gpufort_acc_event_copyin .or. &
              creational_event .eq. gpufort_acc_event_copy ) &
                call records_(loc)%copy_to_device(async)
         !print *, loc
         last_record_index_ = max(loc, last_record_index_)
         total_memory_b_ = total_memory_b_ + blocked_size(num_bytes)
#if DEBUG > 1
         write(output_unit,fmt="(a)",advance="no") "[2]created record: "
         flush(output_unit)
         call records_(loc)%print()
         flush(output_unit)
#endif
      endif
    end function

    subroutine destroy_record_(record)
       use iso_fortran_env
       implicit none
       class(t_record),intent(inout) :: record
       !
       total_memory_b_ = total_memory_b_ - blocked_size(record%num_bytes)
       
#if DEBUG > 1
       write(output_unit,fmt="(a)",advance="no") "[2]destroy record: "
       flush(output_unit)
       call record%print()
       flush(output_unit)
#endif
       
       call record%destroy()
    end subroutine

    !> Deletes a record's reference counter and destroys the record if
    !> the reference counter is zero. Copies the data to the host beforehand
    !> if specified.
    !> 
    !> \note Not thread safe.
    subroutine decrement_delete_record_(hostptr,copy_to_host)
      use iso_c_binding
      use hipfort
      use hipfort_check
      implicit none
      type(c_ptr),intent(in) :: hostptr
      !
      logical :: copy_to_host
      !
      integer :: loc 
      logical :: success
      !
      loc = find_record_(hostptr,success)
      if ( success ) then
         if ( records_(loc)%decrement_num_refs() ) then
           if ( copy_to_host ) call records_(loc)%copy_to_host() 
           call destroy_record_(records_(loc))
         endif
      else
#ifndef DELETE_NORECORD_MEANS_NOOP
        ERROR STOP "gpufort_acc_runtime: decrement_delete_record: could not find matching record for hostptr"
#endif
        return
      endif
      !
    end subroutine
    
    !
    ! public
    !
    subroutine gpufort_acc_init()
      implicit none
      allocate(records_(INITIAL_RECORDS_CAPACITY))
      initialized_ = .TRUE.
    end subroutine
   
    !> Deallocate all host data and the records_ and
    !> queue data structures.. 
    subroutine gpufort_acc_shutdown()
      implicit none
      integer :: i
      if ( .not. initialized_ ) ERROR STOP "gpufort_acc_shutdown: runtime not initialized"
      ! deallocate records_
      do i = 1, last_record_index_
        if ( records_(i)%is_initialized() ) then
          call records_(i)%destroy()
        endif 
      end do
      deallocate(records_)
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
      integer,optional,intent(in),dimension(:) :: arg,async
      !
      integer :: i
      logical :: no_opt_present 
      !
      if ( .not. initialized_ ) ERROR STOP "gpufort_acc_wait: runtime not initialized"
      no_opt_present = .TRUE.
      if ( present(arg) ) then
        no_opt_present = .FALSE.
        do i=1,size(arg)
            call hipCheck(hipStreamSynchronize(queues_(arg(i))%queueptr))
        end do
      endif
      !
      if ( present(async) ) then
        no_opt_present = .FALSE.
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
    subroutine gpufort_acc_enter_region(unstructured)
      implicit none
      logical,optional,intent(in) :: unstructured
      if ( .not. initialized_ ) call gpufort_acc_init()
      current_region_ = current_region_ + 1
    end subroutine
 
    !> \note We can use this also for a "normal" end data section 
    !> \note We can use this also for any other region exit that uses device data
    subroutine gpufort_acc_exit_region(unstructured)
#ifdef EXIT_REGION_SYNCHRONIZE_DEVICE
      use hipfort
      use hipfort_check
#endif
      implicit none
      logical,optional,intent(in) :: unstructured
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
      do i = 1, last_record_index_
        if ( records_(i)%is_initialized() ) then
          if ( records_(i)%region .eq. current_region_ ) then
            if ( records_(i)%creational_event .eq. gpufort_acc_event_create .or.&
                 records_(i)%creational_event .eq. gpufort_acc_event_copyin ) then
              call destroy_record_(records_(i))
            else if ( records_(i)%creational_event .eq. gpufort_acc_event_copyout .or. &
                 records_(i)%creational_event .eq. gpufort_acc_event_copy ) then
              call records_(i)%copy_to_host() ! synchronous
              call destroy_record_(records_(i))
            endif
          else
            new_last_record_index = i
          endif
        endif 
      end do
      last_record_index_ = new_last_record_index
      !
      current_region_ = current_region_ -1
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
    function gpufort_acc_present_b(hostptr,num_bytes,async,copy,copyin) result(deviceptr)
      use iso_fortran_env
      use iso_c_binding
      use gpufort_acc_runtime_c_bindings
      implicit none
      type(c_ptr),intent(in)       :: hostptr
      integer(c_size_t),intent(in) :: num_bytes
      !logical,optional,intent(in) :: exiting
      integer,optional,intent(in)  :: async
      logical,optional,intent(in)  :: copy
      logical,optional,intent(in)  :: copyin
      !
      type(c_ptr) :: deviceptr
      !
      logical :: success,fits
      integer :: loc
      integer(c_size_t) :: offset_bytes
      logical :: opt_copyin
      logical :: opt_copy
      !
      if ( .not. initialized_ ) ERROR STOP "gpufort_acc_present_b: runtime not initialized"
      if ( .not. c_associated(hostptr) ) then
        ERROR STOP "gpufort_acc_present_b: hostptr not c_associated"
        deviceptr = c_null_ptr
      else
        loc = find_record_(hostptr,success)
        if ( .not. success ) then 
           opt_copyin = .FALSE.
           opt_copy   = .FALSE.
           if ( present(copy) )   opt_copy   = copy
           if ( present(copyin) ) opt_copyin = copyin
           if ( opt_copy ) then
             deviceptr = gpufort_acc_copy_b(hostptr,num_bytes,async)
           else if ( opt_copyin ) then
             deviceptr = gpufort_acc_copyin_b(hostptr,num_bytes,async)
           else
             print *, "ERROR: did not find record for hostptr:"
             CALL print_cptr(hostptr)
             ERROR STOP "gpufort_acc_present_b: no record found for hostptr"
           end if
        endif
        ! TODO replace 0 by size of searched hostptr
        fits      = is_subarray(records_(loc)%hostptr,records_(loc)%num_bytes,hostptr,0_8,offset_bytes)
        deviceptr = inc_cptr(records_(loc)%deviceptr,offset_bytes)
#if DEBUG > 0
        write(output_unit,fmt="(a)",advance="no") "[1]gpufort_acc_present_b: retrieved deviceptr="
        flush(output_unit)
        CALL print_cptr(deviceptr)
        flush(output_unit)
        write(output_unit,fmt="(a,i0.0,a)",advance="no") "(offset_bytes=",offset_bytes,",base deviceptr="
        flush(output_unit)
        CALL print_cptr(records_(loc)%deviceptr)
        flush(output_unit)
        write(output_unit,fmt="(a,i0.0)",advance="no") ",base num_bytes=", records_(loc)%num_bytes
        write(output_unit,fmt="(a)",advance="no") ") for hostptr="
        flush(output_unit)
        CALL print_cptr(hostptr)
        flush(output_unit)
        print *, ""
#endif
        !if (exiting) then 
        !  records_(loc)%decrement_num_refs()
        !else
        !  records_(loc)%increment_num_refs()
        !endif
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
    function gpufort_acc_create_b(hostptr,num_bytes) result(deviceptr)
      use iso_c_binding
      implicit none
      type(c_ptr),intent(in)       :: hostptr
      integer(c_size_t),intent(in) :: num_bytes
      type(c_ptr) :: deviceptr
      !
      integer :: loc
      !
      if ( .not. initialized_ ) ERROR STOP "gpufort_acc_create_b: runtime not initialized"
      if ( .not. c_associated(hostptr) ) then
        ERROR STOP "gpufort_acc_create_b: hostptr not c_associated"
        deviceptr = c_null_ptr
      else
        loc = insert_increment_record_(hostptr,num_bytes,gpufort_acc_event_create)
        deviceptr = records_(loc)%deviceptr
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
    function gpufort_acc_no_create_b(hostptr) result(deviceptr)
      use iso_c_binding
      implicit none
      type(c_ptr), intent(in) :: hostptr
      type(c_ptr) :: deviceptr
      !
      integer :: loc
      logical :: success
      !
      if ( .not. initialized_ ) ERROR STOP "gpufort_acc_no_create_b: runtime not initialized"
      if ( .not. c_associated(hostptr) ) then
        ERROR STOP "gpufort_acc_no_create_b: hostptr not c_associated"
        deviceptr = c_null_ptr
      else
        loc = find_record_(hostptr,success)
        deviceptr = c_null_ptr
        if ( success ) deviceptr = records_(loc)%deviceptr
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
    subroutine gpufort_acc_delete_b(hostptr,finalize)
      use iso_c_binding
      implicit none
      type(c_ptr), intent(in)     :: hostptr
      logical,intent(in),optional :: finalize
      !
      integer :: loc 
      logical :: success, opt_finalize 
      !
      if ( .not. initialized_ ) ERROR STOP "gpufort_acc_delete_b: runtime not initialized"
      if ( .not. c_associated(hostptr) ) then
#ifndef NULLPTR_MEANS_NOOP
        ERROR STOP "gpufort_acc_delete_b: hostptr not c_associated"
#endif
        return
      else
        opt_finalize = .FALSE.
        if ( present(finalize) ) opt_finalize = finalize
        if ( opt_finalize ) then
          loc = find_record_(hostptr,success)
          if ( .not. success ) ERROR STOP "gpufort_acc_delete: no record found for hostptr"
          call records_(loc)%destroy()
        else
          call decrement_delete_record_(hostptr,copy_to_host=.FALSE.)
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
    function gpufort_acc_copyin_b(hostptr,num_bytes,async) result(deviceptr)
      use iso_c_binding
      implicit none
      type(c_ptr), intent(in)      :: hostptr
      integer(c_size_t),intent(in) :: num_bytes
      integer,optional,intent(in)  :: async
      !
      type(c_ptr) :: deviceptr 
      !
      integer :: loc
      !
      if ( .not. initialized_ ) ERROR STOP "gpufort_acc_copyin_b: runtime not initialized"
      if ( .not. c_associated(hostptr) ) then
#ifndef NULLPTR_MEANS_NOOP
        ERROR STOP "gpufort_acc_copyin_b: hostptr not c_associated"
#endif
        deviceptr = c_null_ptr
      else
        loc = insert_increment_record_(hostptr,num_bytes,gpufort_acc_event_copyin,async)
        deviceptr = records_(loc)%deviceptr
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
    function gpufort_acc_copyout_b(hostptr,num_bytes,async) result(deviceptr)
      use iso_c_binding
      implicit none
      ! Return type(c_ptr)
      type(c_ptr), intent(in)      :: hostptr
      integer(c_size_t),intent(in) :: num_bytes
      integer,optional,intent(in)  :: async
      !
      type(c_ptr) :: deviceptr 
      !
      integer :: loc
      !
      if ( .not. initialized_ ) ERROR STOP "gpufort_acc_copyout_b: runtime not initialized"
      if ( .not. c_associated(hostptr) ) then
#ifndef NULLPTR_MEANS_NOOP
        ERROR STOP "gpufort_acc_copyout_b: hostptr not c_associated"
#endif
        deviceptr = c_null_ptr
      else
        loc = insert_increment_record_(hostptr,num_bytes,gpufort_acc_event_copyout,async)
        deviceptr = records_(loc)%deviceptr
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
    function gpufort_acc_copy_b(hostptr,num_bytes,async) result(deviceptr)
      use iso_c_binding
      implicit none
      ! Return type(c_ptr)
      type(c_ptr), intent(in)      :: hostptr
      integer(c_size_t),intent(in) :: num_bytes
      integer,optional,intent(in)  :: async
      !
      type(c_ptr) :: deviceptr 
      !
      integer :: loc
      !
      if ( .not. initialized_ ) ERROR STOP "gpufort_acc_copy_b: runtime not initialized"
      if ( .not. c_associated(hostptr) ) then
#ifndef NULLPTR_MEANS_NOOP
        ERROR STOP "gpufort_acc_copy_b: hostptr not c_associated"
#endif
        deviceptr = c_null_ptr
      else
        loc = insert_increment_record_(hostptr,num_bytes,gpufort_acc_event_copy,async)
        deviceptr = records_(loc)%deviceptr
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
    subroutine gpufort_acc_update_host_b(hostptr,condition,if_present,async)
      use iso_c_binding
      implicit none
      type(c_ptr),intent(in)      :: hostptr
      logical,intent(in),optional :: condition, if_present 
      integer,intent(in),optional :: async
      !
      integer :: loc
      logical :: success, opt_condition, opt_if_present 
      !
      if ( .not. initialized_ ) ERROR STOP "gpufort_acc_update_host_b: runtime not initialized"
      if ( .not. c_associated(hostptr) ) then
#ifndef NULLPTR_MEANS_NOOP
        ERROR STOP "gpufort_acc_update_host_b: hostptr not c_associated"
#endif
        return
      endif
      opt_condition  = .True.
      opt_if_present = .False.
      if ( present(condition) )  opt_condition  = condition
      if ( present(if_present) ) opt_if_present = if_present
      !
      if ( opt_condition ) then
        loc = find_record_(hostptr,success)
        !
        if ( .not. success .and. .not. opt_if_present ) ERROR STOP "gpufort_acc_update_host_b: no deviceptr found for hostptr"
        ! 
        if ( success .and. present(async) ) then 
          call records_(loc)%copy_to_host(async)
        else if ( success ) then
          call records_(loc)%copy_to_host()
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
    !>   When the condition is zero or .FALSE., no data will be moved to
    !>   or from the device.
    !> if_present
    !>   Issue no error when the data is not present on the device.
    !> async [( expression )]
    !>   The data movement will execute asynchronously with the
    !>   encountering thread on the corresponding async queue.
    !> wait [( expression-list )] - TODO not implemented
    !>   The data movement will not begin execution until all actions on
    !>   the corresponding async queue(s) are complete.
    subroutine gpufort_acc_update_device_b(hostptr,condition,if_present,async)
      use iso_c_binding
      implicit none
      type(c_ptr),intent(in)      :: hostptr
      logical,intent(in),optional :: condition, if_present 
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
      opt_condition  = .True.
      opt_if_present = .False.
      if ( present(condition) )  opt_condition  = condition
      if ( present(if_present) ) opt_if_present = if_present
      !
      if ( opt_condition ) then
        loc = find_record_(hostptr,success)
        !
        if ( .not. success .and. .not. opt_if_present ) ERROR STOP "gpufort_acc_update_device_b: no deviceptr found for hostptr"
        ! 
        if ( success .and. present(async) ) then 
          call records_(loc)%copy_to_device(async)
        else if ( success ) then
          call records_(loc)%copy_to_device()
        endif
      endif
    end subroutine

end module
