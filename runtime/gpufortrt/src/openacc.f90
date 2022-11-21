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
    enumerator :: acc_device_hip 
    enumerator :: acc_device_radeon 
    enumerator :: acc_device_nvidia 
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

  interface acc_copyin
    module procedure :: acc_copyin_nb
    module procedure :: acc_copyin_b
  end interface
  interface acc_copyin_async
    module procedure :: acc_copyin_async_nb
    module procedure :: acc_copyin_async_b
  end interface

  interface acc_copyout
    module procedure :: acc_copyout_nb
    module procedure :: acc_copyout_b
  end interface
  interface acc_copyout_async
    module procedure :: acc_copyout_async_nb
    module procedure :: acc_copyout_async_b
  end interface
  interface acc_copyout_finalize
    module procedure :: acc_copyout_finalize_nb
    module procedure :: acc_copyout_finalize_b
  end interface
  interface acc_copyout_finalize_async
    module procedure :: acc_copyout_finalize_async_nb
    module procedure :: acc_copyout_finalize_async_b
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

  integer(c_size_t) function acc_get_property(dev_num, dev_type, property) 
    use iso_c_binding
    implicit none
    !
    integer(kind=acc_device_kind),value, intent(in) :: dev_type
    integer,value, intent(in) :: dev_num
    integer(kind=acc_property_kind),value, intent(in) :: property
    interface
      integer(c_size_t) function acc_get_property_c_impl(dev_num, dev_type, property) &
        bind(c,name="acc_get_property_f")
        use iso_c_binding
        Import::acc_device_kind, acc_property_kind
        implicit none
        !
        integer(kind=acc_device_kind), value, intent(in):: dev_type
        integer(c_int),value, intent(in) :: dev_num
        integer(kind=acc_property_kind), value, intent(in):: property
      end function
    end interface
    acc_get_property = acc_get_property_c_impl(dev_num, dev_type, property)
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
    integer,value,intent(in) :: bytes
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
    acc_is_present_nb = acc_is_present_b(c_loc(data_arg),int(sizeof(data_arg)))
  end function

  subroutine acc_wait(wait_arg)
    use iso_c_binding
    implicit none
    !
    integer(acc_handle_kind),dimension(..),target,intent(in),optional :: wait_arg
    !
    interface
      subroutine acc_wait_c_impl(wait_arg) &
        bind(c,name="acc_wait")
        use iso_c_binding
        implicit none
        !
        type(c_ptr),value,intent(in) :: wait_arg
        !
      end subroutine
    end interface
    !
    if ( present(wait_arg) ) then
        call acc_wait_c_impl(c_loc(wait_arg))
      endif
  end subroutine

  subroutine acc_wait_async(wait_arg, async_arg)
    use iso_c_binding
    implicit none
    !
    integer(acc_handle_kind),dimension(..),target,intent(in),optional :: wait_arg, async_arg
    !
    interface
      subroutine acc_wait_async_c_impl(wait_arg, async_arg) &
        bind(c,name="acc_wait_async")
        use iso_c_binding
        implicit none
        !
        type(c_ptr),value,intent(in) :: wait_arg, async_arg
        !
      end subroutine
    end interface
    !
    if ( present(wait_arg) ) then
      if ( present(async_arg) ) then
        call acc_wait_async_c_impl(c_loc(wait_arg), c_loc(async_arg))
      endif
    endif
  end subroutine

  subroutine acc_wait_all()
    use iso_c_binding
    implicit none
    !
    interface
      subroutine acc_wait_all_c_impl() &
        bind(c,name="acc_wait_all")
        use iso_c_binding
        implicit none
        !
      end subroutine
    end interface
    !
    call acc_wait_all_c_impl()
  end subroutine

  subroutine acc_wait_all_async(async_arg)
    use iso_c_binding
    implicit none
    !
    integer(acc_handle_kind),dimension(..),target,intent(in),optional :: async_arg
    !
    interface
      subroutine acc_wait_all_async_c_impl(async_arg) &
        bind(c,name="acc_wait_all_async")
        use iso_c_binding
        implicit none
        !
        type(c_ptr),value,intent(in) :: async_arg
        !
      end subroutine
    end interface
    !
    if ( present(async_arg) ) then
      call acc_wait_all_async_c_impl(c_loc(async_arg))
    endif
  end subroutine

  subroutine acc_wait_device(wait_arg, dev_num)
    use iso_c_binding
    implicit none
    !
    integer(acc_handle_kind),dimension(..),target,intent(in),optional :: wait_arg
    integer(c_int), optional :: dev_num
    !
    interface
      subroutine acc_wait_device_c_impl(wait_arg, dev_num) &
        bind(c,name="acc_wait_device")
        use iso_c_binding
        implicit none
        !
        type(c_ptr),value,intent(in) :: wait_arg
        integer(c_int),value,intent(in) :: dev_num
        !
      end subroutine
    end interface
    !
    if ( present(wait_arg) ) then
        if( present(dev_num) ) then
          call acc_wait_device_c_impl(c_loc(wait_arg), dev_num)
        endif
    endif
  end subroutine

  subroutine acc_wait_device_async(wait_arg, async_arg, dev_num)
    use iso_c_binding
    implicit none
    !
    integer(acc_handle_kind),dimension(..),target,intent(in),optional :: wait_arg, async_arg
    integer(c_int), optional :: dev_num
    !
    interface
      subroutine acc_wait_device_async_c_impl(wait_arg, async_arg, dev_num) &
        bind(c,name="acc_wait_device_async")
        use iso_c_binding
        implicit none
        !
        type(c_ptr),value,intent(in) :: wait_arg, async_arg
        integer(c_int),value,intent(in) :: dev_num
        !
      end subroutine
    end interface
    !
    if ( present(wait_arg) ) then
      if ( present(async_arg) ) then
        if( present(dev_num) ) then
          call acc_wait_device_async_c_impl(c_loc(wait_arg), c_loc(async_arg), dev_num)
        endif
      endif
    endif
  end subroutine

  subroutine acc_wait_all_device(dev_num)
    use iso_c_binding
    implicit none
    !
    integer(c_int), optional :: dev_num
    !
    interface
      subroutine acc_wait_all_device_c_impl(dev_num) &
        bind(c,name="acc_wait_all_device")
        use iso_c_binding
        implicit none
        !
        integer(c_int),value,intent(in) :: dev_num
        !
      end subroutine
    end interface
    !
    if( present(dev_num) ) then
      call acc_wait_all_device_c_impl(dev_num)
    endif
  end subroutine

  subroutine acc_wait_all_device_async(async_arg, dev_num)
    use iso_c_binding
    implicit none
    !
    integer(acc_handle_kind),dimension(..),target,intent(in),optional :: async_arg
    integer(c_int), optional :: dev_num
    !
    interface
      subroutine acc_wait_all_device_async_c_impl(async_arg, dev_num) &
        bind(c,name="acc_wait_all_device_async")
        use iso_c_binding
        implicit none
        !
        type(c_ptr),value,intent(in) :: async_arg
        integer(c_int),value,intent(in) :: dev_num
        !
      end subroutine
    end interface
    !
    if ( present(async_arg) ) then
      if( present(dev_num) ) then
        call acc_wait_all_device_async_c_impl(c_loc(async_arg), dev_num)
      endif
    endif
  end subroutine

  logical function acc_async_test(wait_arg)
    use iso_c_binding
    implicit none
    !
    integer(acc_handle_kind),dimension(..),target,intent(in) :: wait_arg
    !
    interface
      integer(c_int) function acc_async_test_c_impl(wait_arg) &
        bind(c,name="acc_async_test")
        use iso_c_binding
        implicit none
        !
        type(c_ptr),value,intent(in) :: wait_arg
        !
      end function
    end interface
    !
    acc_async_test = acc_async_test_c_impl(c_loc(wait_arg)) > 0
  end function

  logical function acc_async_test_device(wait_arg, dev_num)
    use iso_c_binding
    implicit none
    !
    integer(acc_handle_kind),dimension(..),target,intent(in) :: wait_arg
    integer(c_int), optional :: dev_num
    !
    interface
      integer(c_int) function acc_async_test_device_c_impl(wait_arg, dev_num) &
        bind(c,name="acc_async_test_device")
        use iso_c_binding
        implicit none
        !
        type(c_ptr),value,intent(in) :: wait_arg
        integer(c_int), optional :: dev_num
        !
      end function
    end interface
    !
    acc_async_test_device = acc_async_test_device_c_impl(c_loc(wait_arg), dev_num) > 0
  end function

  logical function acc_async_test_all()
    use iso_c_binding
    implicit none
    !
    interface
      integer(c_int) function acc_async_test_all_c_impl() &
        bind(c,name="acc_async_test_all")
        use iso_c_binding
        implicit none
        !
      end function
    end interface
    !
    acc_async_test_all = acc_async_test_all_c_impl() > 0
  end function

  logical function acc_async_test_all_device(dev_num)
    use iso_c_binding
    implicit none
    !
    integer(c_int), value, intent(in) :: dev_num
    !
    interface
      integer(c_int) function acc_async_test_all_device_c_impl(dev_num) &
        bind(c,name="acc_async_test_all_device")
        use iso_c_binding
        implicit none
        !
        integer(c_int), value, intent(in) :: dev_num
        !
      end function
    end interface
    !
    acc_async_test_all_device = acc_async_test_all_device_c_impl(dev_num) > 0
  end function

  subroutine acc_copyin_b(data_arg, bytes)
    use iso_c_binding
    implicit none
    !
    type(*), target, dimension(..)::data_arg
    integer,value,intent(in) :: bytes
    interface 
      subroutine acc_copyin_b_c_impl(data_arg,bytes) &
      bind(c,name="acc_copyin")
        use iso_c_binding
        implicit none
        !
        type(c_ptr), value::data_arg
        integer(c_size_t), value :: bytes
      end subroutine
    end interface
    call acc_copyin_b_c_impl(c_loc(data_arg),int(bytes,kind=c_size_t))
  end subroutine

  subroutine acc_copyin_nb(data_arg)
    use iso_c_binding
    implicit none
    !
    type(*), target, dimension(..)::data_arg
    call acc_copyin_b(data_arg, int(sizeof(data_arg)))
  end subroutine

  subroutine acc_copyin_async_b(data_arg, bytes, async_arg)
    use iso_c_binding
    implicit none
    !
    type(*), target, dimension(..)::data_arg
    integer,value,intent(in) :: bytes
    integer(acc_handle_kind),dimension(..),target,intent(in) :: async_arg
    interface 
      subroutine acc_copyin_async_b_c_impl(data_arg,bytes, async_arg) &
      bind(c,name="acc_copyin_async")
        use iso_c_binding
        implicit none
        !
        type(c_ptr), value::data_arg
        integer(c_size_t), value :: bytes
        type(c_ptr),value,intent(in) :: async_arg
      end subroutine
    end interface
    call acc_copyin_async_b_c_impl(c_loc(data_arg),int(bytes,kind=c_size_t), c_loc(async_arg))
  end subroutine

  subroutine acc_copyin_async_nb(data_arg, async_arg)
    use iso_c_binding
    implicit none
    !
    type(*), target, dimension(..)::data_arg
    integer(acc_handle_kind),dimension(..),target,intent(in) :: async_arg
    call acc_copyin_async_b(data_arg,int(sizeof(data_arg)), async_arg)
  end subroutine

  subroutine acc_copyout_b(data_arg, bytes)
    use iso_c_binding
    implicit none
    !
    type(*), target, dimension(..)::data_arg
    integer,value,intent(in) :: bytes
    interface 
      subroutine acc_copyout_b_c_impl(data_arg,bytes) &
      bind(c,name="acc_copyout")
        use iso_c_binding
        implicit none
        !
        type(c_ptr), value::data_arg
        integer(c_size_t), value :: bytes
      end subroutine
    end interface
    call acc_copyout_b_c_impl(c_loc(data_arg),int(bytes,kind=c_size_t))
  end subroutine

  subroutine acc_copyout_nb(data_arg)
    use iso_c_binding
    implicit none
    !
    type(*), target, dimension(..)::data_arg
    call acc_copyout_b(data_arg, int(sizeof(data_arg)))
  end subroutine

  subroutine acc_copyout_async_b(data_arg, bytes, async_arg)
    use iso_c_binding
    implicit none
    !
    type(*), target, dimension(..)::data_arg
    integer,value,intent(in) :: bytes
    integer(acc_handle_kind),dimension(..),target,intent(in) :: async_arg
    interface 
      subroutine acc_copyout_async_b_c_impl(data_arg,bytes, async_arg) &
      bind(c,name="acc_copyout_async")
        use iso_c_binding
        implicit none
        !
        type(c_ptr), value::data_arg
        integer(c_size_t), value :: bytes
        type(c_ptr),value,intent(in) :: async_arg
      end subroutine
    end interface
    call acc_copyout_async_b_c_impl(c_loc(data_arg),int(bytes,kind=c_size_t), c_loc(async_arg))
  end subroutine

  subroutine acc_copyout_async_nb(data_arg, async_arg)
    use iso_c_binding
    implicit none
    !
    type(*), target, dimension(..)::data_arg
    integer(acc_handle_kind),dimension(..),target,intent(in) :: async_arg
    call acc_copyout_async_b(data_arg,int(sizeof(data_arg)), async_arg)
  end subroutine

  subroutine acc_copyout_finalize_b(data_arg, bytes)
    use iso_c_binding
    implicit none
    !
    type(*), target, dimension(..)::data_arg
    integer,value,intent(in) :: bytes
    interface 
      subroutine acc_copyout_finalize_b_c_impl(data_arg,bytes) &
      bind(c,name="acc_copyout_finalize")
        use iso_c_binding
        implicit none
        !
        type(c_ptr), value::data_arg
        integer(c_size_t), value :: bytes
      end subroutine
    end interface
    call acc_copyout_finalize_b_c_impl(c_loc(data_arg),int(bytes,kind=c_size_t))
  end subroutine

  subroutine acc_copyout_finalize_nb(data_arg)
    use iso_c_binding
    implicit none
    !
    type(*), target, dimension(..)::data_arg
    call acc_copyout_finalize_b(data_arg, int(sizeof(data_arg)))
  end subroutine

  subroutine acc_copyout_finalize_async_b(data_arg, bytes, async_arg)
    use iso_c_binding
    implicit none
    !
    type(*), target, dimension(..)::data_arg
    integer,value,intent(in) :: bytes
    integer(acc_handle_kind),dimension(..),target,intent(in) :: async_arg
    interface 
      subroutine acc_copyout_finalize_async_b_c_impl(data_arg,bytes, async_arg) &
      bind(c,name="acc_copyout_finalize_async")
        use iso_c_binding
        implicit none
        !
        type(c_ptr), value::data_arg
        integer(c_size_t), value :: bytes
        type(c_ptr),value,intent(in) :: async_arg
      end subroutine
    end interface
    call acc_copyout_finalize_async_b_c_impl(c_loc(data_arg),int(bytes,kind=c_size_t), c_loc(async_arg))
  end subroutine

  subroutine acc_copyout_finalize_async_nb(data_arg, async_arg)
    use iso_c_binding
    implicit none
    !
    type(*), target, dimension(..)::data_arg
    integer(acc_handle_kind),dimension(..),target,intent(in) :: async_arg
    call acc_copyout_finalize_async_b(data_arg,int(sizeof(data_arg)), async_arg)
  end subroutine

end module
