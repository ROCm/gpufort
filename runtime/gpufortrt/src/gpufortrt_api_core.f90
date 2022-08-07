! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
module gpufortrt_core
  use iso_c_binding
  use gpufortrt_types

  public

contains

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
    use iso_c_binding
    use auxiliary
    implicit none
    !
    !integer,intent(in)                              :: device_kind
    type(mapping_t),dimension(:),intent(in),optional :: mappings
    integer,intent(in),optional                      :: async
    logical,intent(in),optional                      :: finalize
    !
    interface
      subroutine gpufortrt_enter_exit_data_c_impl(mappings,num_mappings,finalize) bind(c,name="gpufortrt_enter_exit_data")
        use iso_c_binding
        implicit none
        type(c_ptr),value,intent(in)    :: mappings
        integer(c_int),value,intent(in) :: num_mappings
        logical(c_bool),value,intent(in) :: finalize
      end subroutine
      subroutine gpufortrt_enter_exit_data_async_c_impl(mappings,num_mappings,async,finalize) bind(c,name="gpufortrt_enter_exit_data_async")
        use iso_c_binding
        implicit none
        type(c_ptr),value,intent(in)    :: mappings
        integer(c_int),value,intent(in) :: num_mappings
        integer(c_int),value,intent(in) :: async
        logical(c_bool),value,intent(in) :: finalize
      end subroutine
    end interface 
    !
    if ( present(async) ) then
      if ( present(mappings) ) then
        call gpufortrt_enter_exit_data_async_c_impl(&
          c_loc(mappings),&
          size(mappings,kind=c_int_t),&
          async,&
          logical(eval_optval(finalize,.false.),kind=c_bool)) 
      else
        call gpufortrt_enter_exit_data_async_c_impl(&
          c_null_ptr,&
          0_c_size_t,&
          async,&
          logical(eval_optval(finalize,.false.),kind=c_bool)) 
      endif
    else
      if ( present(mappings) ) then
        call gpufortrt_enter_exit_data_c_impl(&
          c_loc(mappings),&
          size(mappings,kind=c_int_t),&
          logical(eval_optval(finalize,.false.),kind=c_bool)) 
      else
        call gpufortrt_enter_exit_data_c_impl(&
          c_null_ptr,&
          0_c_int_t,&
          logical(eval_optval(finalize,.false.),kind=c_bool)) 
      endif
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
        function gpufortrt_use_device_c_impl(hostptr,num_bytes,condition,if_present) &
            bind(c,name="gpufortrt_use_device") &
            result(deviceptr)
          type(c_ptr),value,intent(in)       :: hostptr
          integer(c_size_t),value,intent(in) :: num_bytes
          logical(c_bool),value,intent(in)   :: condition, if_present
        end function
      end interface
      resultptr = gpufortrt_use_device_c_impl(hostptr,num_bytes,&
          eval_optval(condition,.false._c_bool),&
          eval_optval(if_present,.false._c_bool),&
    end function
    ! TODO rank-bound procedures via jinja template

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

    function gpufortrt_present_b(hostptr,num_bytes,update_struct_refs) result(deviceptr)
      use iso_fortran_env
      use iso_c_binding
      implicit none
      type(c_ptr),intent(in)       :: hostptr
      integer(c_size_t),intent(in) :: num_bytes
      type(gpufortrt_counter_t),optional,intent(in)  :: ctr_to_update
      !
      type(c_ptr) :: deviceptr
      !
      interface
        subroutine gpufortrt_present_c_impl(hostptr,num_bytes,ctr_to_update) &
            bind(c,name="gpufortrt_present") result(deviceptr)
          type(c_ptr),value,intent(in)                :: hostptr
          integer(c_size_t),value,intent(in)          :: num_bytes
          type(gpufortrt_counter_t),value,intent(in)  :: ctr_to_update
        end subroutine
      end interface
      call gpufortrt_present_c_impl(hostptr,num_bytes,&
          eval_optval(ctr_to_update,gpufortrt_counter_none))
    end function

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

    function gpufortrt_create_b(hostptr,num_bytes,async,never_deallocate,&
                                ctr_to_update) result(deviceptr)
      use iso_c_binding
      implicit none
      type(c_ptr),intent(in)       :: hostptr
      integer(c_size_t),intent(in) :: num_bytes
      integer,intent(in),optional  :: async ! ignored as there is no hipMallocAsync
      logical,intent(in),optional  :: never_deallocate
      type(gpufortrt_counter_t),optional,intent(in)  :: ctr_to_update
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

    subroutine gpufortrt_delete_b(hostptr,finalize)
      use iso_c_binding
      implicit none
      type(c_ptr), intent(in)     :: hostptr
      logical,intent(in),optional :: finalize
    end subroutine

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
    end function

    function gpufortrt_copyout_b(hostptr,num_bytes,async,&
            update_struct_refs,update_dyn_refs) result(deviceptr)
      use iso_c_binding
      implicit none
      ! Return type(c_ptr)
      type(c_ptr), intent(in)      :: hostptr
      integer(c_size_t),intent(in) :: num_bytes
      integer,intent(in),optional  :: async
      logical,intent(in),optional  :: update_struct_refs,update_dyn_refs
    end function

    function gpufortrt_copy_b(hostptr,num_bytes,async) &
        result(deviceptr)
      use iso_c_binding
      implicit none
      ! Return type(c_ptr)
      type(c_ptr), intent(in)                                      :: hostptr
      integer(c_size_t),intent(in)                                 :: num_bytes
      integer(c_int),intent(in),optional                           :: async
      integer(kind(gpufortrt_counter_none)),intent(in),optional  :: ctr_to_update
      !
      type(c_ptr) :: deviceptr
      !
      interface
        function gpufortrt_copy_c_impl(hostptr,num_bytes,async,ctr_to_update) &
          result(deviceptr) bind(c,name="gpufortrt_copy")
          type(c_ptr),value,intent(in)                              :: hostptr
          integer(c_size_t),value,intent(in)                        :: num_bytes
          integer(c_int),value,intent(in)                           :: async
          integer(kind(gpufortrt_counter_none)),value,intent(in)  :: ctr_to_update
          !
          type(c_ptr) :: deviceptr
        end function
      end interface
      call gpufortrt_copy_c_impl(
        hostptr,num_bytes,

    end function

    subroutine gpufortrt_update_self_b(hostptr,num_bytes,condition,if_present,async)
      use iso_c_binding
      use auxiliary
      implicit none
      type(c_ptr),intent(in)                :: hostptr
      integer(c_size_t),intent(in),optional :: num_bytes
      logical(c_bool),intent(in),optional   :: condition, if_present
      integer(c_int),intent(in),optional    :: async
      !
      interface
        subroutine gpufortrt_update_self_c_impl(hostptr,condition,if_present) &
                bind(c,name="gpufortrt_update_host")
          type(c_ptr),value,intent(in)     :: hostptr
          logical(c_bool),value,intent(in) :: condition, if_present
        end subroutine
        subroutine gpufortrt_update_self_section_c_impl(hostptr,num_bytes,condition,if_present) &
                bind(c,name="gpufortrt_update_self_section")
          type(c_ptr),value,intent(in)       :: hostptr
          integer(c_size_t),value,intent(in) :: num_bytes
          logical(c_bool),value,intent(in)   :: condition, if_present
        end subroutine
        subroutine gpufortrt_update_self_async_c_impl(hostptr,condition,if_present,async) &
                bind(c,name="gpufortrt_update_self_async")
          type(c_ptr),value,intent(in)     :: hostptr
          logical(c_bool),value,intent(in) :: condition, if_present
          integer(c_int),value,intent(in)  :: async
        end subroutine
        subroutine gpufortrt_update_self_section_async_c_impl(hostptr,num_bytes,condition,if_present,async) &
                bind(c,name="gpufortrt_update_self_section_async")
          type(c_ptr),value,intent(in)       :: hostptr
          integer(c_size_t),value,intent(in) :: num_bytes
          logical(c_bool),value,intent(in)   :: condition, if_present
          integer(c_int),value,intent(in)    :: async
        end subroutine
      end interface
      if ( present(num_bytes) ) then
        if ( present(async) ) then
          call gpufortrt_update_self_section_async_c_impl(hostptr,&
                                                          num_bytes,&
                                                          eval_optval(condition,.false._c_bool),&
                                                          eval_optval(if_present,.false._c_bool),&
                                                          async)
        else
          call gpufortrt_update_self_section_c_impl(hostptr,&
                                                    num_bytes,&
                                                    eval_optval(condition,.false._c_bool),&
                                                    eval_optval(if_present,.false._c_bool),&
                                                    gpufortrt_async_noval)
        endif
      else
        if ( present(async) ) then
          call gpufortrt_update_self_async_c_impl(hostptr,&
                                                  eval_optval(condition,.false._c_bool),&
                                                  eval_optval(if_present,.false._c_bool),&
                                                  async)
        else
          call gpufortrt_update_self_c_impl(hostptr,&
                                            eval_optval(condition,.false._c_bool),&
                                            eval_optval(if_present,.false._c_bool),&
                                            gpufortrt_async_noval)
        endif
      endif
    end subroutine

    !> Update Directive
    subroutine gpufortrt_update_device_b(hostptr,num_bytes,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_auxiliary
      implicit none
      type(c_ptr),intent(in)                :: hostptr
      integer(c_size_t),intent(in),optional :: num_bytes
      logical(c_bool),intent(in),optional   :: condition, if_present
      integer(c_int),intent(in),optional    :: async
      !
      interface
        subroutine gpufortrt_update_device_c_impl(hostptr,condition,if_present) &
                bind(c,name="gpufortrt_update_host")
          type(c_ptr),value,intent(in)     :: hostptr
          logical(c_bool),value,intent(in) :: condition, if_present
        end subroutine
        subroutine gpufortrt_update_device_section_c_impl(hostptr,num_bytes,condition,if_present) &
                bind(c,name="gpufortrt_update_device_section")
          type(c_ptr),value,intent(in)       :: hostptr
          integer(c_size_t),value,intent(in) :: num_bytes
          logical(c_bool),value,intent(in)   :: condition, if_present
        end subroutine
        subroutine gpufortrt_update_device_async_c_impl(hostptr,condition,if_present,async) &
                bind(c,name="gpufortrt_update_device_async")
          type(c_ptr),value,intent(in)     :: hostptr
          logical(c_bool),value,intent(in) :: condition, if_present
          integer(c_int),value,intent(in)  :: async
        end subroutine
        subroutine gpufortrt_update_device_section_async_c_impl(hostptr,num_bytes,condition,if_present,async) &
                bind(c,name="gpufortrt_update_device_section_async")
          type(c_ptr),value,intent(in)       :: hostptr
          integer(c_size_t),value,intent(in) :: num_bytes
          logical(c_bool),value,intent(in)   :: condition, if_present
          integer(c_int),value,intent(in)    :: async
        end subroutine
      end interface
      if ( present(num_bytes) ) then
        if ( present(async) ) then
          call gpufortrt_update_device_section_async_c_impl(hostptr,&
                                                          num_bytes,&
                                                          eval_optval(condition,.false._c_bool),&
                                                          eval_optval(if_present,.false._c_bool),&
                                                          async)
        else
          call gpufortrt_update_device_section_c_impl(hostptr,&
                                                    num_bytes,&
                                                    eval_optval(condition,.false._c_bool),&
                                                    eval_optval(if_present,.false._c_bool),&
                                                    gpufortrt_async_noval)
        endif
      else
        if ( present(async) ) then
          call gpufortrt_update_device_async_c_impl(hostptr,&
                                                  eval_optval(condition,.false._c_bool),&
                                                  eval_optval(if_present,.false._c_bool),&
                                                  async)
        else
          call gpufortrt_update_device_c_impl(hostptr,&
                                            eval_optval(condition,.false._c_bool),&
                                            eval_optval(if_present,.false._c_bool),&
                                            gpufortrt_async_noval)
        endif
      endif
    end subroutine

    function gpufortrt_get_stream(queue_id) result (stream)
       implicit none
       integer, intent(in) :: queue_id
       !
       type(c_ptr) :: stream
       !
    end function
end module
