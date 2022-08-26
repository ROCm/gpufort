! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
module openacc
  use iso_c_binding, only: c_int
  private :: c_int
  
  integer, parameter :: acc_handle_kind = c_int

  integer(acc_handle_kind), parameter :: acc_async_noval = -1 
  integer(acc_handle_kind), parameter :: acc_async_sync = -2 
  integer(acc_handle_kind), parameter :: acc_async_default = -3 

  integer, parameter :: acc_device_kind = kind(acc_device_none)
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
  interface
    subroutine acc_set_device_type(dev_type) bind(c,name="acc_set_device_type")
      implicit none
      integer(acc_device_kind),value :: dev_type
    end subroutine
  
    integer(acc_device_kind) function acc_get_device_type() bind(c,name="acc_get_device_type")
      implicit none
    end function

    function acc_get_property(dev_num, dev_type, property) bind(c,name="acc_get_property")
      use iso_c_binding, only: c_size_t
      implicit none
      integer, value :: dev_num
      integer(acc_device_kind), value :: dev_type
      integer(acc_device_property_kind), value :: property
      integer(c_size_t) :: acc_get_property
    end function

    subroutine acc_init(dev_type) bind(c,name="acc_init")
      implicit none
      integer(acc_device_kind),value :: dev_type
    end subroutine

    subroutine acc_shutdown(dev_type) bind(c,name="acc_shutdown")
      implicit none
      integer(acc_device_kind),value :: dev_type
    end subroutine

    subroutine acc_wait(wait_arg) bind(c,name="acc_wait")
      implicit none
      integer(acc_handle_kind),value :: wait_arg
    end subroutine

    subroutine acc_wait_all() bind(c,name="acc_wait_all")
      implicit none
    end subroutine
    
    subroutine acc_wait_all_async(wait_arg) bind(c,name="acc_wait_all_async")
      implicit none
      integer(acc_handle_kind),value :: wait_arg
    end subroutine

    function acc_get_default_async() bind(c,name="acc_get_default_async")
      implicit none
      integer(acc_handle_kind),value :: acc_get_default_async
    end function

    subroutine acc_set_default_async(async_arg) bind(c,name="acc_set_default_async")
      implicit none
      integer(acc_handle_kind),value :: acc_get_default_async
    end subroutine 

  end interface

  ! Overloaded routines 
  
  interface acc_copyin
    module procedure :: acc_copyin_nb
    module procedure :: acc_copyin_b
  end interface
  
  interface acc_copyin_async
    module procedure :: acc_copyin_async_nb
    module procedure :: acc_copyin_async_b
  end interface
  
  interface acc_create
    module procedure :: acc_create_nb
    module procedure :: acc_create_b
  end interface
  
  interface acc_create_async
    module procedure :: acc_create_async_nb
    module procedure :: acc_create_async_b
  end interface

  interface acc_update_device
    module procedure :: acc_update_device_nb
    module procedure :: acc_update_device_b
  end interface
  
  interface acc_update_device_async
    module procedure :: acc_update_device_async_nb
    module procedure :: acc_update_device_async_b
  end interface
  
  interface acc_update_self
    module procedure :: acc_update_self_nb
    module procedure :: acc_update_self_b
  end interface
  
  interface acc_update_self_async
    module procedure :: acc_update_self_async_nb
    module procedure :: acc_update_self_async_b
  end interface
  
  interface acc_is_present
    module procedure :: acc_is_present_nb
    module procedure :: acc_is_present_b
  end interface

  ! Legacy interfaces
  interface acc_present_or_copyin
    module procedure :: acc_copyin_nb
    module procedure :: acc_copyin_b
  end interface
  interface acc_pcopyin
    module procedure :: acc_copyin_nb
    module procedure :: acc_copyin_b
  end interface
  
  interface acc_present_or_create
    module procedure :: acc_create_nb
    module procedure :: acc_create_b
  end interface
  interface acc_pcreate
    module procedure :: acc_create_nb
    module procedure :: acc_create_b
  end interface

  ! Non-standard extensions

  interface acc_malloc
    type(c_ptr) function acc_malloc_c_impl(bytes) bind(c,name="acc_malloc")
      use iso_c_binding, only: c_size_t, c_ptr
      implicit none
      integer(c_size_t),value :: bytes
    end function
  end interface
  
  interface acc_free
    subroutine acc_free_c_impl(data_dev) bind(c,name="acc_free")
      use iso_c_binding, only: c_ptr
      implicit none
      type(c_ptr),value :: data_dev
    end subroutine
  end interface

contains
  integer function acc_get_num_devices(dev_type)
    use iso_c_binding
    implicit none
    interface
      integer(c_int) function acc_get_num_devices_c_impl(dev_type) bind(c,name="acc_get_num_devices")
        implicit none
        integer(acc_device_kind),value :: dev_type
      end function
    end interface
    acc_get_num_devices = int(acc_get_num_devices_c_impl(dev_type))
  end function
    
  subroutine acc_set_device_num(dev_num, dev_type)
    implicit none
    integer :: dev_num
    integer(acc_device_kind) :: dev_type
    interface   
      subroutine acc_set_device_num_c_impl(dev_num, dev_type) bind(c,name="acc_set_device_num")
        implicit none
         integer(c_int),value :: dev_num
         integer(acc_device_kind),value :: dev_type
      end subroutine         
    end interface
    call acc_set_device_num_c_impl(int(dev_num,kind=c_int), dev_type)
  end subroutine         

  integer function acc_get_device_num(dev_type)
    implicit none
    integer(acc_device_kind) :: dev_type
    interface
      integer(c_int) function acc_get_device_num_c_impl(dev_type) bind(c,name="acc_get_device_num")
        implicit none
      end function
    end interface
    acc_get_device_num = int(acc_get_device_num_c_impl(dev_type))
  end function

  subroutine acc_get_property_string(dev_num, dev_type,&
    implicit none
                                     property, string)
    integer, value :: dev_num
    integer(acc_device_kind), value :: dev_type
    integer(acc_device_property_kind), value :: property
    character*(*) :: string
    ! TODO 
    !interface
    !  subroutine acc_get_property_string_c_impl(dev_num,dev_type,property,string) bind(c,name="acc_get_property_string")
    !    use iso_c_binding, only: c_int
    !    integer(c_int), value :: dev_num
    !    integer(acc_device_kind), value :: dev_type
    !    integer(acc_device_property_kind), value :: property
    !    character*(kind=c_char,len=*) :: string
    !  end subroutine
    !end interface
    !character(kind=c_char,len=*) :: 
    !call acc_get_property_string_c_impl(dev_num,dev_type,property,string)
    ERROR STOP "acc_get_property_string not implemented yet!"
  end subroutine

  logical function acc_async_test(wait_arg)
    implicit none
    integer(acc_handle_kind) :: wait_arg
    interface
      integer(c_int) function acc_async_test_c_impl(wait_arg) bind(c,name="acc_async_test")
        implicit none
        integer(acc_handle_kind),value :: wait_arg
      end function
    end interface
    acc_async_test = acc_async_test_c_impl(wait_arg) > 0
  end function
  
  logical function acc_async_test_device(wait_arg, dev_num)
    implicit none
    integer(acc_handle_kind) :: wait_arg
    integer :: dev_num
    interface
      integer(c_int) function acc_async_test_device_c_impl(wait_arg, dev_num) &
              bind(c,name="acc_async_test_device")
        use iso_c_binding, only: c_int
        implicit none
        integer(acc_handle_kind),value :: wait_arg
        integer(c_int),value :: dev_num
      end function
    end interface
    acc_async_test_device = acc_async_test_device_c_impl(wait_arg,int(dev_num,kind=c_int)) > 0
  end function
  
  logical function acc_async_test_all()
    implicit none
    interface
      integer(c_int) function acc_async_test_all_c_impl() &
        bind(c,name=acc_async_test_all)
        implicit none
      end function
    end interface
    acc_async_test_all = acc_async_test_all_c_impl() > 0
  end function
  
  logical function acc_async_test_all_device(dev_num)
    implicit none
    integer :: dev_num
    interface 
      integer(c_int) function acc_async_test_all_device_c_impl(dev_num) bind(c,name="acc_async_test_all_device")
        use iso_c_binding, only: c_int
        implicit none
        integer(c_int),value :: dev_num
      end function
    end interface
    acc_async_test_device = acc_async_test_all_device(int(dev_num,kind=c_int)) > 0
  end function
  
  subroutine acc_wait_device(wait_arg, dev_num)
    implicit none
    integer(acc_handle_kind) :: wait_arg
    integer :: dev_num
    interface
      subroutine acc_wait_device_c_impl(wait_arg, dev_num) &
              bind(c,name="acc_wait_device")
        use iso_c_binding, only: c_int
        implicit none
        integer(acc_handle_kind),value :: wait_arg
        integer(c_int),value :: dev_num
      end subroutine
    end interface
    call acc_wait_device_c_impl(wait_arg,int(dev_num,kind=c_int))
  end subroutine
  
  subroutine acc_wait_async(wait_arg, async_arg)
    implicit none
    integer(acc_handle_kind) :: wait_arg, async_arg
    interface
      subroutine acc_wait_async_c_impl(wait_arg, async_arg) &
              bind(c,name="acc_wait_async")
                implicit none
        integer(acc_handle_kind),value :: wait_arg, async_arg
      end subroutine
    end interface
    call acc_wait_async_c_impl(wait_arg,async_arg)
  end subroutine
  
  subroutine acc_wait_device_async(wait_arg, async_arg, dev_num)
    implicit none
    integer(acc_handle_kind) :: wait_arg, async_arg
    integer :: dev_num
    interface
      subroutine acc_wait_device_async_c_impl(wait_arg, async_arg, dev_num) &
              bind(c,name="acc_wait_device_async")
        use iso_c_binding, only: c_int
        implicit none
        integer(acc_handle_kind),value :: wait_arg, async_arg
        integer(c_int),value :: dev_num
      end subroutine
    end interface
    call acc_wait_device_async_c_impl(wait_arg,async_arg,int(dev_num,kind=c_int))
  end subroutine
  
  subroutine acc_wait_all_device(dev_num)
    implicit none
    integer :: dev_num
    interface
      subroutine acc_wait_all_device_c_impl(dev_num) &
              bind(c,name="acc_wait_all_device")
        use iso_c_binding, only: c_int
        implicit none
        integer(c_int),value :: dev_num
      end subroutine
    end interface
    call acc_wait_all_device_c_impl(int(dev_num,kind=c_int))
  end subroutine
  
  subroutine acc_wait_all_device_async(async_arg, dev_num)
    implicit none
    integer(acc_handle_kind) :: async_arg
    integer :: dev_num
    interface
      subroutine acc_wait_all_device_async_c_impl(async_arg, dev_num) &
              bind(c,name="acc_wait_all_device_async")
        use iso_c_binding, only: c_int
        implicit none
        integer(acc_handle_kind),value :: async_arg
        integer(c_int),value :: dev_num
      end subroutine
    end interface
    call acc_wait_all_device_async_c_impl(async_arg,int(dev_num,kind=c_int))
  end subroutine

  logical function acc_on_device(dev_type)
    implicit none
    integer(acc_device_kind) :: dev_type
    interface 
      integer(c_int) function acc_on_device_c_impl(dev_type) bind(c,name="acc_on_device")
        use iso_c_binding, only: c_int
        implicit none
        integer(acc_device_kind),value :: dev_type
      end function
    end interface
    acc_on_device = acc_on_device_c_impl(dev_type) > 0
  end function

  !subroutine acc_copyin(data_arg) # TODO
  !subroutine acc_copyin_async(data_arg, async_arg) # TODO
  subroutine acc_copyin(data_arg, bytes)
    implicit none
    type(*), target, dimension(..) :: data_arg
    integer :: bytes
    !
    type(c_ptr) :: tmp
    interface
      function acc_copyin_c_impl(data_arg, bytes) bind(c,name="acc_copyin")
        implicit none
        type(c_ptr),value :: hostptr
        integer(c_size_t),value :: num_bytes
        type(c_ptr) :: acc_copyin_c_impl
      end function
    end interface
    tmp =  acc_copyin_c_impl(c_loc(data_arg), int(num_bytes,kind=c_size_t))
  end subroutine
  subroutine acc_copyin_async(data_arg, bytes, async_arg)
    implicit none
    type(*), dimension(..) :: data_arg
    integer :: bytes
    integer(acc_handle_kind) :: async_arg
    interface
      subroutine acc_copyin_async_c_impl(data_arg, bytes) bind(c,name="acc_copyin_async")
        implicit none
        type(c_ptr),value :: hostptr
        integer(c_size_t),value :: num_bytes
        integer(acc_handle_kind),value :: async_arg
      end subroutine
    end interface
    call acc_copyin_async_c_impl(c_loc(data_arg), int(num_bytes,kind=c_size_t))
  end subroutine
  
  !subroutine acc_create(data_arg) # TODO
  !subroutine acc_create_async(data_arg, async_arg) # TODO
  subroutine acc_create(data_arg, bytes)
    implicit none
    type(*), target, dimension(..) :: data_arg
    integer :: bytes
    !
    type(c_ptr) :: tmp
    interface
      function acc_create_c_impl(data_arg, bytes) bind(c,name="acc_create")
        implicit none
        type(c_ptr),value :: hostptr
        integer(c_size_t),value :: num_bytes
        type(c_ptr) :: acc_create_c_impl
      end function
    end interface
    tmp =  acc_create_c_impl(c_loc(data_arg), int(num_bytes,kind=c_size_t))
  end subroutine
  subroutine acc_create_async(data_arg, bytes, async_arg)
    implicit none
    type(*), dimension(..) :: data_arg
    integer :: bytes
    integer(acc_handle_kind) :: async_arg
    interface
      subroutine acc_create_async_c_impl(data_arg, bytes) bind(c,name="acc_create_async")
        implicit none
        type(c_ptr),value :: hostptr
        integer(c_size_t),value :: num_bytes
        integer(acc_handle_kind),value :: async_arg
      end subroutine
    end interface
    call acc_create_async_c_impl(c_loc(data_arg), int(num_bytes,kind=c_size_t))
  end subroutine
  
  !subroutine acc_copyout(data_arg) # TODO
  !subroutine acc_copyout_async(data_arg, async_arg) # TODO
  subroutine acc_copyout(data_arg, bytes)
    implicit none
    type(*), target, dimension(..) :: data_arg
    integer :: bytes
    interface
      subroutine acc_copyout_c_impl(data_arg, bytes) bind(c,name="acc_copyout")
        implicit none
        type(c_ptr),value :: hostptr
        integer(c_size_t),value :: num_bytes
      end subroutine
    end interface
    call acc_copyout_c_impl(c_loc(data_arg), int(num_bytes,kind=c_size_t))
  end subroutine
  subroutine acc_copyout_async(data_arg, bytes, async_arg)
    implicit none
    type(*), dimension(..) :: data_arg
    integer :: bytes
    integer(acc_handle_kind) :: async_arg
    interface
      subroutine acc_copyout_async_c_impl(data_arg, bytes) bind(c,name="acc_copyout_async")
        implicit none
        type(c_ptr),value :: hostptr
        integer(c_size_t),value :: num_bytes
        integer(acc_handle_kind),value :: async_arg
      end subroutine
    end interface
    call acc_copyout_async_c_impl(c_loc(data_arg), int(num_bytes,kind=c_size_t))
  end subroutine
  !subroutine acc_copyout_finalize(data_arg) # TODO
  !subroutine acc_copyout_finalize_async(data_arg, async_arg) # TODO
  subroutine acc_copyout_finalize(data_arg, bytes)
    implicit none
    type(*), target, dimension(..) :: data_arg
    integer :: bytes
    interface
      subroutine acc_copyout_finalize_c_impl(data_arg, bytes) bind(c,name="acc_copyout_finalize")
        implicit none
        type(c_ptr),value :: hostptr
        integer(c_size_t),value :: num_bytes
      end subroutine
    end interface
    call acc_copyout_finalize_c_impl(c_loc(data_arg), int(num_bytes,kind=c_size_t))
  end subroutine
  subroutine acc_copyout_finalize_async(data_arg, bytes, async_arg)
    implicit none
    type(*), dimension(..) :: data_arg
    integer :: bytes
    integer(acc_handle_kind) :: async_arg
    interface
      subroutine acc_copyout_finalize_async_c_impl(data_arg, bytes) bind(c,name="acc_copyout_finalize_async")
        implicit none
        type(c_ptr),value :: hostptr
        integer(c_size_t),value :: num_bytes
        integer(acc_handle_kind),value :: async_arg
      end subroutine
    end interface
    call acc_copyout_finalize_async_c_impl(c_loc(data_arg), int(num_bytes,kind=c_size_t))
  end subroutine
  
  !subroutine acc_delete(data_arg) # TODO
  !subroutine acc_delete_async(data_arg, async_arg) # TODO
  subroutine acc_delete(data_arg, bytes)
    implicit none
    type(*), target, dimension(..) :: data_arg
    integer :: bytes
    interface
      subroutine acc_delete_c_impl(data_arg, bytes) bind(c,name="acc_delete")
        implicit none
        type(c_ptr),value :: hostptr
        integer(c_size_t),value :: num_bytes
      end subroutine
    end interface
    call acc_delete_c_impl(c_loc(data_arg), int(num_bytes,kind=c_size_t))
  end subroutine
  subroutine acc_delete_async(data_arg, bytes, async_arg)
    implicit none
    type(*), dimension(..) :: data_arg
    integer :: bytes
    integer(acc_handle_kind) :: async_arg
    interface
      subroutine acc_delete_async_c_impl(data_arg, bytes) bind(c,name="acc_delete_async")
        implicit none
        type(c_ptr),value :: hostptr
        integer(c_size_t),value :: num_bytes
        integer(acc_handle_kind),value :: async_arg
      end subroutine
    end interface
    call acc_delete_async_c_impl(c_loc(data_arg), int(num_bytes,kind=c_size_t))
  end subroutine
  !subroutine acc_delete_finalize(data_arg) # TODO
  !subroutine acc_delete_finalize_async(data_arg, async_arg) # TODO
  subroutine acc_delete_finalize(data_arg, bytes)
    implicit none
    type(*), target, dimension(..) :: data_arg
    integer :: bytes
    interface
      subroutine acc_delete_finalize_c_impl(data_arg, bytes) bind(c,name="acc_delete_finalize")
        implicit none
        type(c_ptr),value :: hostptr
        integer(c_size_t),value :: num_bytes
      end subroutine
    end interface
    call acc_delete_finalize_c_impl(c_loc(data_arg), int(num_bytes,kind=c_size_t))
  end subroutine
  subroutine acc_delete_finalize_async(data_arg, bytes, async_arg)
    implicit none
    type(*), dimension(..) :: data_arg
    integer :: bytes
    integer(acc_handle_kind) :: async_arg
    interface
      subroutine acc_delete_finalize_async_c_impl(data_arg, bytes) bind(c,name="acc_delete_finalize_async")
        implicit none
        type(c_ptr),value :: hostptr
        integer(c_size_t),value :: num_bytes
        integer(acc_handle_kind),value :: async_arg
      end subroutine
    end interface
    call acc_delete_finalize_async_c_impl(c_loc(data_arg), int(num_bytes,kind=c_size_t))
  end subroutine

  !subroutine acc_update_device(data_arg) # TODO
  !subroutine acc_update_device_async(data_arg, async_arg) TODO
  subroutine acc_update_device(data_arg, bytes)
    implicit none
    type(*), dimension(..) :: data_arg
    integer :: bytes
    interface
      subroutine acc_update_device_c_impl(data_arg, bytes) bind(c,name="acc_update_device")
        implicit none
        type(c_ptr),value :: data_arg
        integer(c_size_t),value :: bytes
      end subroutine
    end interface
    call acc_update_device_c_impl(c_loc(data_arg),int(bytes,kind=c_size_t))
  end subroutine

  subroutine acc_update_device_async(data_arg, bytes, async_arg)
    implicit none
    type(*), dimension(..) :: data_arg
    integer :: bytes
    integer(acc_handle_kind) :: async_arg
    interface
      subroutine acc_update_device_c_impl(data_arg, bytes) bind(c,name="acc_update_device")
        implicit none
        type(c_ptr),value :: data_arg
        integer(c_size_t),value :: bytes
        integer(acc_handle_kind),value :: async_arg
      end subroutine
    end interface
    call acc_update_device_c_impl(c_loc(data_arg),int(bytes,kind=c_size_t),async_arg)
  end subroutine
  
  subroutine acc_update_self_b(data_arg, bytes)
    implicit none
    type(*), dimension(..) :: data_arg
    integer :: bytes
    interface
      subroutine acc_update_self_c_impl(data_arg, bytes) bind(c,name="acc_update_self")
        implicit none
        type(c_ptr),value :: data_arg
        integer(c_size_t),value :: bytes
      end subroutine
    end interface
    call acc_update_self_c_impl(c_loc(data_arg),int(bytes,kind=c_size_t))
  end subroutine

  subroutine acc_update_self_async_b(data_arg, bytes, async_arg)
    implicit none
    type(*), dimension(..) :: data_arg
    integer :: bytes
    integer(acc_handle_kind) :: async_arg
    interface
      subroutine acc_update_self_c_impl(data_arg, bytes) bind(c,name="acc_update_self")
        implicit none
        type(c_ptr),value :: data_arg
        integer(c_size_t),value :: bytes
        integer(acc_handle_kind),value :: async_arg
      end subroutine
    end interface
    call acc_update_self_c_impl(c_loc(data_arg),int(bytes,kind=c_size_t),async_arg)
  end subroutine
  
  subroutine acc_update_self_b(data_arg) 
    implicit none
    type(*), dimension(..), contiguous :: data_arg
    call acc_update_self_b(data_arg,sizeof(data_arg))
  end subroutine
  subroutine acc_update_self_async_nb(data_arg, async_arg)
    implicit none
    type(*), dimension(..) :: data_arg
    integer(acc_handle_kind) :: async_arg
    call acc_update_self_async_b(data_arg,sizeof(data_arg),async_arg)
  end subroutine

  logical function acc_is_present_b(data_arg, bytes)
    implicit none
    type(*), dimension(..) :: data_arg
    integer :: bytes
    interface 
      integer(c_int) function acc_is_present_c_impl(data_arg,bytes) bind(c,name="acc_is_present")
        implicit none
        type(c_ptr),value :: data_arg
        integer(c_size_t),value :: bytes
      end function
    end interface
    acc_is_present_b = acc_is_present_c_impl(c_loc(data_arg),int(bytes,kind=c_size_t)) > 0
  end function
 
  logical function acc_is_present_nb(data_arg)
    implicit none
    type(*), dimension(..), contiguous :: data_arg
    acc_is_present_nb = acc_is_present_b(data_arg,sizeof(data_arg)) 
  end function
end module
