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
end module
