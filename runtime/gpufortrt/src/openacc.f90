! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
module openacc
  use iso_c_binding, only: c_int
  private :: c_int
  
  integer, parameter :: acc_handle_kind = c_int

  integer(acc_handle_kind), parameter :: acc_async_noval = -1 
  integer(acc_handle_kind), parameter :: acc_async_sync = -2 
  integer(acc_handle_kind), parameter :: acc_async_default = -3 

  enum, bind(c)
    enumerator :: acc_device_none = 0
    enumerator :: acc_device_default
    enumerator :: acc_device_host
    enumerator :: acc_device_not_host
    enumerator :: acc_device_current
    enumerator :: acc_device_hip = acc_device_not_host
    enumerator :: acc_device_radeon = acc_device_hip
    enumerator :: acc_device_nvidia = acc_device_hip
  end enum
  
  integer, parameter :: acc_device_kind = kind(acc_device_none)
  
  enum, bind(c)
    enumerator :: acc_property_memory = 0 !>integer,  size of device memory in bytes
    enumerator :: acc_property_free_memory !>integer,  free device memory in bytes
    enumerator :: acc_property_shared_memory_support !>integer,  nonzero if the specified device supports sharing memory with the local thread
    enumerator :: acc_property_name !>string, device name*/
    enumerator :: acc_property_vendor !>string, device vendor*/
    enumerator :: acc_property_driver !>string, device driver version*/
  end enum

  integer, parameter :: acc_property_kind = kind(acc_property_memory)

  ! direct delegation to C implementation

  interface acc_is_present
    module procedure :: acc_is_present_nb
    module procedure :: acc_is_present_b
  end interface

contains

  subroutine acc_init(dev_type) 
    use iso_c_binding
    implicit none
    !
    integer(kind=acc_device_kind),value :: dev_type
      interface
        subroutine acc_init_c_impl(dev_type) &
          bind(c,name="acc_init")
          use iso_c_binding
          Import::acc_device_kind
          implicit none
          !
          integer(kind=acc_device_kind), value, intent(in):: dev_type
        end subroutine
      end interface
      call acc_init_c_impl(dev_type)
  end subroutine

  subroutine acc_shutdown(dev_type) 
    use iso_c_binding
    implicit none
    !
    integer(kind=acc_device_kind),value :: dev_type
      interface
        subroutine acc_shutdown_c_impl(dev_type) &
          bind(c,name="acc_shutdown")
          use iso_c_binding
          Import::acc_device_kind
          implicit none
          !
          integer(kind=acc_device_kind), value, intent(in):: dev_type
        end subroutine
      end interface
      call acc_shutdown_c_impl(dev_type)
  end subroutine

  subroutine acc_set_device_type(dev_type) 
      use iso_c_binding
      implicit none
      !
      integer(kind=acc_device_kind),value :: dev_type
      interface
        subroutine acc_set_device_type_c_impl(dev_type) &
          bind(c,name="acc_set_device_type")
          use iso_c_binding
          Import::acc_device_kind
          implicit none
          !
          integer(kind=acc_device_kind), value, intent(in):: dev_type
        end subroutine
      end interface
      call acc_set_device_type_c_impl(dev_type)
  end subroutine

  integer(kind=acc_device_kind) function acc_get_device_type() 
    use iso_c_binding
    implicit none
    !
    interface
    integer(c_int) function acc_get_device_type_c_impl() &
        bind(c,name="acc_get_device_type")
        use iso_c_binding
        implicit none
        !
      end function
    end interface
    acc_get_device_type = int(acc_get_device_type_c_impl(),kind=acc_device_kind)
  end function

  subroutine acc_set_device_num(dev_num, dev_type) 
    use iso_c_binding
    implicit none
    !
    integer(kind=acc_device_kind),value, intent(in) :: dev_type
    integer,value, intent(in) :: dev_num
    interface
      subroutine acc_set_device_num_c_impl(dev_num, dev_type) &
        bind(c,name="acc_set_device_num_f")
        use iso_c_binding
        Import::acc_device_kind
        implicit none
        !
        integer(kind=acc_device_kind), value, intent(in):: dev_type
        integer(c_int),value, intent(in) :: dev_num
      end subroutine
    end interface
    call acc_set_device_num_c_impl(dev_num, dev_type)
  end subroutine

  integer function acc_get_device_num(dev_type)
    use iso_c_binding
    implicit none
    !
    integer(kind=acc_device_kind),value :: dev_type
    interface
      integer(c_int) function acc_get_device_num_c_impl(dev_type) &
        bind(c,name="acc_get_device_num_f")
        use iso_c_binding  
        Import::acc_device_kind
        implicit none
        !
        integer(kind=acc_device_kind),value :: dev_type
      end function
    end interface
    acc_get_device_num = int(acc_get_device_num_c_impl(dev_type))
  end function

  integer function acc_get_num_devices(dev_type)
    use iso_c_binding
    implicit none
    !
    integer(kind=acc_device_kind),value :: dev_type
    interface
      integer(c_int) function acc_get_num_devices_c_impl(dev_type) bind(c,name="acc_get_num_devices")
        use iso_c_binding  
        Import::acc_device_kind
        implicit none
        !
        integer(kind=acc_device_kind),value :: dev_type
      end function
    end interface
    acc_get_num_devices = int(acc_get_num_devices_c_impl(dev_type))
  end function
  
  logical function acc_is_present_b(data_arg, bytes)
    use iso_c_binding
    implicit none
    !
    type(*), target, dimension(..)::data_arg
    integer(c_int),value,intent(in) :: bytes
    interface 
      integer(c_int) function acc_is_present_c_impl(data_arg,bytes) &
        bind(c,name="acc_is_present")
        use iso_c_binding
        implicit none
        !
        type(c_ptr), value::data_arg
        integer(c_size_t), value :: bytes
      end function
    end interface
    acc_is_present_b = acc_is_present_c_impl(c_loc(data_arg),int(bytes,kind=c_size_t)) > 0
  end function
 
  logical function acc_is_present_nb(data_arg)
    use iso_c_binding
    implicit none
    !
    type(*), target, dimension(..), contiguous :: data_arg
    acc_is_present_nb = acc_is_present_b(c_loc(data_arg),size(data_arg))
  end function

  subroutine acc_wait_device(wait_arg, dev_num, condition)
    use iso_c_binding
    implicit none
    !
    integer(acc_handle_kind),dimension(..),target,intent(in),optional :: wait_arg
    integer(c_int), optional :: dev_num
    logical(c_bool),intent(in), optional :: condition
    !
    interface
      subroutine acc_wait_device_c_impl(wait_arg, num_wait_args, dev_num, condition) &
        bind(c,name="acc_wait_device")
        use iso_c_binding
        implicit none
        !
        type(c_ptr),value,intent(in) :: wait_arg
        integer(c_int),value,intent(in) :: num_wait_args
        integer(c_int),value,intent(in) :: dev_num
        logical(c_bool),value,intent(in) :: condition
        !
      end subroutine
    end interface
    !
    logical(c_bool) :: opt_if_arg
    !
    opt_if_arg = .true._c_bool
    if ( present(condition) ) opt_if_arg = logical(condition,kind=c_bool)
    if ( present(wait_arg) ) then
        if( present(dev_num) ) then
          call acc_wait_device_c_impl(&
          c_loc(wait_arg), size(wait_arg,kind=c_int), dev_num, opt_if_arg)
        endif
    endif
  end subroutine

  subroutine acc_wait_device_async(wait_arg, async_arg, dev_num, condition)
    use iso_c_binding
    implicit none
    !
    integer(acc_handle_kind),dimension(..),target,intent(in),optional :: wait_arg, async_arg
    integer(c_int), optional :: dev_num
    logical(c_bool),value,intent(in), optional :: condition
    !
    interface
      subroutine acc_wait_device_async_c_impl(&
        wait_arg, num_wait_args, &
        async_arg, num_async_args, &
        dev_num, condition) &
        bind(c,name="acc_wait_device_async")
        use iso_c_binding
        implicit none
        !
        type(c_ptr),value,intent(in) :: wait_arg, async_arg
        integer(c_int),value,intent(in) :: num_wait_args, num_async_args
        integer(c_int),value,intent(in) :: dev_num
        logical(c_bool),value,intent(in) :: condition
        !
      end subroutine
    end interface
    !
    logical(c_bool) :: opt_if_arg
    !
    opt_if_arg = .true._c_bool
    if ( present(condition) ) opt_if_arg = logical(condition,kind=c_bool)
    if ( present(wait_arg) ) then
      if ( present(async_arg) ) then
        if( present(dev_num) ) then
          call acc_wait_device_async_c_impl(&
          c_loc(wait_arg), size(wait_arg,kind=c_int),&
          c_loc(async_arg), size(async_arg,kind=c_int), dev_num, opt_if_arg)
        endif
      endif
    endif
  end subroutine

  subroutine acc_wait_all_device(dev_num, condition)
    use iso_c_binding
    implicit none
    !
    integer(c_int), optional :: dev_num
    logical(c_bool),intent(in), optional :: condition
    !
    interface
      subroutine acc_wait_all_device_c_impl(dev_num, condition) &
        bind(c,name="acc_wait_all_device")
        use iso_c_binding
        implicit none
        !
        integer(c_int),value,intent(in) :: dev_num
        logical(c_bool),value,intent(in) :: condition
        !
      end subroutine
    end interface
    !
    logical(c_bool) :: opt_if_arg
    !
    opt_if_arg = .true._c_bool
    if ( present(condition) ) opt_if_arg = logical(condition,kind=c_bool)
      if( present(dev_num) ) then
        call acc_wait_all_device_c_impl(dev_num, opt_if_arg)
      endif
  end subroutine

  subroutine acc_wait_all_device_async(async_arg, dev_num, condition)
    use iso_c_binding
    implicit none
    !
    integer(acc_handle_kind),dimension(..),target,intent(in),optional :: async_arg
    integer(c_int), optional :: dev_num
    logical(c_bool),value,intent(in), optional :: condition
    !
    interface
      subroutine acc_wait_all_device_async_c_impl(&
        async_arg, num_async_args, &
        dev_num, condition) &
        bind(c,name="acc_wait_all_device_async")
        use iso_c_binding
        implicit none
        !
        type(c_ptr),value,intent(in) :: async_arg
        integer(c_int),value,intent(in) :: num_async_args
        integer(c_int),value,intent(in) :: dev_num
        logical(c_bool),value,intent(in) :: condition
        !
      end subroutine
    end interface
    !
    logical(c_bool) :: opt_if_arg
    !
    opt_if_arg = .true._c_bool
    if ( present(condition) ) opt_if_arg = logical(condition,kind=c_bool)
    if ( present(async_arg) ) then
      if( present(dev_num) ) then
        call acc_wait_all_device_async_c_impl(&
        c_loc(async_arg), size(async_arg,kind=c_int), dev_num, opt_if_arg)
      endif
    endif
  end subroutine
end module
