! SPDX-License-Identifier: MIT                                                
! Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
! This module is stateless to prevent data races when OpenMP is enabled.
module openacc_gomp_base
  use iso_c_binding
  ! expose openacc interfaces
  use openacc

  private ! make everything private by default

  public :: goacc_data_start, goacc_data_end, goacc_enter_exit_data
  public :: print_cptr
  public :: mapping_init_

  !< based on: https://raw.githubusercontent.com/gcc-mirror/gcc/41d6b10e96a1de98e90a7c0378437c3255814b16/include/gomp-constants.h
  !< commit 8e8f643 on Jan 3, 2018
  !< always check if this file has been modified!
  public :: GOMP_MAP_ALLOC,GOMP_MAP_TO,GOMP_MAP_FROM,GOMP_MAP_TOFROM,GOMP_MAP_POINTER,GOMP_MAP_TO_PSET,GOMP_MAP_FORCE_PRESENT,GOMP_MAP_DELETE,GOMP_MAP_FORCE_DEVICEPTR,GOMP_MAP_DEVICE_RESIDENT,GOMP_MAP_LINK,GOMP_MAP_FIRSTPRIVATE,GOMP_MAP_FIRSTPRIVATE_INT,GOMP_MAP_USE_DEVICE_PTR,GOMP_MAP_ZERO_LEN_ARRAY_SECTION,GOMP_MAP_FORCE_ALLOC,GOMP_MAP_FORCE_TO,GOMP_MAP_FORCE_FROM,GOMP_MAP_FORCE_TOFROM,GOMP_MAP_ALWAYS_TO,GOMP_MAP_ALWAYS_FROM,GOMP_MAP_ALWAYS_TOFROM,GOMP_MAP_STRUCT,GOMP_MAP_ALWAYS_POINTER,GOMP_MAP_DELETE_ZERO_LEN_ARRAY_SECTION,GOMP_MAP_RELEASE,GOMP_MAP_FIRSTPRIVATE_POINTER,GOMP_MAP_FIRSTPRIVATE_REFERENCE

  enum, bind(c)
    enumerator :: GOMP_MAP_ALLOC = 0
    enumerator :: GOMP_MAP_TO = 1
    enumerator :: GOMP_MAP_FROM = 2
    enumerator :: GOMP_MAP_TOFROM = 3
    enumerator :: GOMP_MAP_POINTER = 4
    enumerator :: GOMP_MAP_TO_PSET = 5
    enumerator :: GOMP_MAP_FORCE_PRESENT = 6
    enumerator :: GOMP_MAP_DELETE = 7
    enumerator :: GOMP_MAP_FORCE_DEVICEPTR = 8
    enumerator :: GOMP_MAP_DEVICE_RESIDENT = 9
    enumerator :: GOMP_MAP_LINK = 10
    enumerator :: GOMP_MAP_FIRSTPRIVATE = 12
    enumerator :: GOMP_MAP_FIRSTPRIVATE_INT = 13
    enumerator :: GOMP_MAP_USE_DEVICE_PTR = 14
    enumerator :: GOMP_MAP_ZERO_LEN_ARRAY_SECTION = 15
    enumerator :: GOMP_MAP_FORCE_ALLOC = 128
    enumerator :: GOMP_MAP_FORCE_TO = 129
    enumerator :: GOMP_MAP_FORCE_FROM = 130
    enumerator :: GOMP_MAP_FORCE_TOFROM = 131
    enumerator :: GOMP_MAP_ALWAYS_TO = 17
    enumerator :: GOMP_MAP_ALWAYS_FROM = 18
    enumerator :: GOMP_MAP_ALWAYS_TOFROM = 19
    enumerator :: GOMP_MAP_STRUCT = 28
    enumerator :: GOMP_MAP_ALWAYS_POINTER = 29
    enumerator :: GOMP_MAP_DELETE_ZERO_LEN_ARRAY_SECTION = 31
    enumerator :: GOMP_MAP_RELEASE = 23
    enumerator :: GOMP_MAP_FIRSTPRIVATE_POINTER = 257
    enumerator :: GOMP_MAP_FIRSTPRIVATE_REFERENCE = 258
  end enum

  type, public :: mapping
     type(c_ptr)                   :: hostptr
     integer(c_size_t)             :: num_bytes
     integer(kind(GOMP_MAP_ALLOC)) :: map_kind
     
     contains
       procedure :: init => mapping_init_
  end type

  type :: context
    type(c_ptr),pointer,dimension(:)                   :: vars      => null()
    integer(c_size_t),pointer,dimension(:)             :: num_bytes => null()
    integer(kind(GOMP_MAP_ALLOC)),pointer,dimension(:) :: maps      => null()

    contains
      procedure :: init            => context_init_
      procedure :: destroy         => context_destroy_
  end type
  
  interface 
   
    ! auxiliary

    subroutine print_cptr(ptr) bind(c,name="print_cptr")
      use iso_c_binding
      implicit none
      type(c_ptr),intent(in),value :: ptr
    end subroutine
  
    !< void
    !< GOACC_data_start (int device, size_t mapnum,
    !<    void **hostaddrs, size_t *sizes, unsigned short *kinds)
    !<
    !< \note: Need to introduce additional c layer because backend is expecting array of unsigned short. Unsigned data types are not
    !supported in Fortran. C interface needs to copy data into aray of unsigned shorts.
    recursive subroutine goacc_data_start_internal_(device,mapnum,hostaddrs,sizes,kinds) bind(c,name="GOACC_data_start_wrapper")
      use iso_c_binding
      implicit none
      integer(c_int),value    :: device
      integer(c_size_t),value :: mapnum
      type(c_ptr),value       :: hostaddrs !< pointer/reference -> void**
      type(c_ptr),value       :: sizes     !< pointer/reference -> size_t*
      type(c_ptr),value       :: kinds     !< pointer/reference -> unsigned short* 
    end subroutine
  
    !< void GOACC_data_end (void)
    !< \note: Need to introduce additional c layer because backend is expecting array of unsigned short. Unsigned data types are not
    !supported in Fortran. C interface needs to copy data into aray of unsigned shorts.
    recursive subroutine goacc_data_end_internal_() bind(c,name="GOACC_data_end")
    end subroutine
  
    !< Interface to a C function that finds a way around the variadic last argument. 
    !< void
    !< GOACC_enter_exit_data (int device, size_t mapnum,
    !<          void **hostaddrs, size_t *sizes, unsigned short *kinds,
    !<          int async, int num_waits, ...)
    recursive subroutine goacc_enter_exit_data_internal_(device,mapnum,hostaddrs,sizes,kinds,async,num_waits,waits) bind(c,name="GOACC_enter_exit_data_wrapper")
      use iso_c_binding
      use openacc, only: acc_handle_kind
      implicit none
      integer(c_int),value           :: device
      integer(c_size_t),value        :: mapnum
      type(c_ptr),value              :: hostaddrs !< pointer/reference -> void**
      type(c_ptr),value              :: sizes     !< pointer/reference -> size_t*
      type(c_ptr),value              :: kinds     !< pointer/reference -> unsigned short* 
      integer(acc_handle_kind),value :: async
      integer(c_int),value           :: num_waits
      type(c_ptr),value              :: waits     !< pointer/reference -> int*
    end subroutine
  
  end interface

contains
 
  ! mapping

  recursive subroutine mapping_init_(this,hostptr,num_bytes,map_kind)
     use iso_c_binding
     implicit none
     !
     class(mapping),intent(inout)             :: this
     type(c_ptr)                  ,intent(in) :: hostptr
     integer(c_size_t)            ,intent(in) :: num_bytes
     integer(kind(GOMP_MAP_ALLOC)),intent(in) :: map_kind
     !
     this%hostptr   = hostptr
     this%num_bytes = num_bytes
     this%map_kind  = map_kind
  end subroutine

  ! context
  
  recursive subroutine context_init_(this,mappings)
    use iso_c_binding
    implicit none
    !
    class(context),intent(inout)          :: this
    type(mapping),dimension(:),intent(in) :: mappings
    !
    integer :: num_maps,i
    !
    num_maps = size(mappings) 
    !
    if ( num_maps > 0 ) then
      allocate(this%vars(num_maps))
      allocate(this%num_bytes(num_maps))
      allocate(this%maps(num_maps))
      ! unpack mapping structs
      do i = 1, size(mappings)
        this%vars(i)  = mappings(i)%hostptr
        this%num_bytes(i) = mappings(i)%num_bytes
        this%maps(i)  = mappings(i)%map_kind
      end do
    endif
  end subroutine

  recursive subroutine context_destroy_(this)
    implicit none
    !
    class(context),intent(inout) :: this
    !
    deallocate(this%vars,this%num_bytes,this%maps)
  end subroutine

  ! public interfaces

  recursive subroutine goacc_data_start(device,mappings,async)
    use iso_c_binding
    use openacc
    implicit none
    !
    integer,intent(in)                             :: device
    type(mapping),dimension(:),optional,intent(in) :: mappings
    !
    type(c_ptr)                                         :: opt_hostaddrs, opt_sizes, opt_kinds, opt_waits
    integer(acc_handle_kind),optional,target,intent(in) :: async ! do nothing with this opt arg
    integer(c_size_t)                                   :: opt_mapnum
    integer                                             :: i
    type(context)                                       :: ctx
    ! mappings
    opt_mapnum = 0
    opt_hostaddrs = c_null_ptr
    opt_sizes     = c_null_ptr
    opt_kinds     = c_null_ptr
    if ( present(mappings) ) opt_mapnum = size(mappings)
    if ( opt_mapnum > 0 ) then
      call ctx%init(mappings)
      opt_hostaddrs = c_loc(ctx%vars(1))
      !do i = 1,opt_mapnum
      !  call print_cptr(ctx%vars(i))
      !  Print *,""
      !end do
      opt_sizes     = c_loc(ctx%num_bytes(1))
      opt_kinds     = c_loc(ctx%maps(1))
    endif
    
    call goacc_data_start_internal_(device,opt_mapnum,opt_hostaddrs,opt_sizes,opt_kinds)
    
    if ( opt_mapnum > 0 ) call ctx%destroy()
  end subroutine

  recursive subroutine goacc_data_end()
    use iso_c_binding
    use openacc
    implicit none
    !
    call goacc_data_end_internal_()
  end subroutine

  recursive subroutine goacc_enter_exit_data(device,mappings,async,waits)
    use iso_c_binding
    use openacc, only: acc_handle_kind, acc_async_sync
    implicit none
    !
    integer,intent(in)                                     :: device
    type(mapping),dimension(:),optional,target,intent(in)  :: mappings
    integer(acc_handle_kind),optional,target,intent(in)    :: async
    integer(c_int),dimension(:),optional,target,intent(in) :: waits
    !
    type(c_ptr)              :: opt_hostaddrs, opt_sizes, opt_kinds, opt_waits
    integer(acc_handle_kind) :: opt_async
    integer(c_size_t)        :: opt_mapnum
    integer(c_int)           :: opt_num_waits
    integer                  :: i
    type(context)            :: ctx
    ! mappings
    opt_mapnum      = 0
    opt_hostaddrs   = c_null_ptr
    opt_sizes       = c_null_ptr
    opt_kinds       = c_null_ptr
    if ( present(mappings) ) opt_mapnum = size(mappings)
    if ( opt_mapnum > 0 ) then
      call ctx%init(mappings)
      opt_hostaddrs = c_loc(ctx%vars(1))
      opt_sizes     = c_loc(ctx%num_bytes(1))
      opt_kinds     = c_loc(ctx%maps(1))
      !do i = 1,opt_mapnum
      !  call print_cptr(ctx%vars(i))
      !  Print *,""
      !end do
    endif
    ! async
    opt_async = acc_async_sync
    if ( present(async) ) then 
      opt_async = async      
    endif 
    ! waits
    opt_num_waits = 0
    opt_waits     = c_null_ptr
    if ( present(waits) ) then 
      opt_num_waits = size(waits)
      if ( opt_num_waits > 0 ) then
        opt_waits = c_loc(waits(1))
      endif
    endif
    ! 
    call goacc_enter_exit_data_internal_(device,opt_mapnum,opt_hostaddrs,opt_sizes,opt_kinds,&
            opt_async,opt_num_waits,opt_waits)
    !
    if ( opt_mapnum > 0 ) call ctx%destroy()
  end subroutine

end module