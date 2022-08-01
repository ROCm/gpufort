! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.

! TODO not thread-safe, make use of OMP_LIB module if this becomes a problem

! TODO will be slow for large pointer storages
! If this becomes an issue, use faster data structures or bind the public interfaces to
! a faster C runtime


module gpufortrt
  use gpufortrt_core
    
  !> Lookup device pointer for given host pointer.
  !> \param[in] condition condition that must be met, otherwise host pointer is returned. Defaults to '.true.'.
  !> \param[in] if_present Only return device pointer if one could be found for the host pointer.
  !>                       otherwise host pointer is returned. Defaults to '.false.'.
  !> \note Returns a c_null_ptr if the host pointer is invalid, i.e. not C associated.
  interface gpufortrt_use_device
    module procedure &
      gpufortrt_use_device_b, &
      gpufortrt_use_device_l_0,&
      gpufortrt_use_device_l_1,&
      gpufortrt_use_device_l_2,&
      gpufortrt_use_device_l_3,&
      gpufortrt_use_device_l_4,&
      gpufortrt_use_device_l_5,&
      gpufortrt_use_device_l_6,&
      gpufortrt_use_device_l_7,&
      gpufortrt_use_device_c_0,&
      gpufortrt_use_device_c_1,&
      gpufortrt_use_device_c_2,&
      gpufortrt_use_device_c_3,&
      gpufortrt_use_device_c_4,&
      gpufortrt_use_device_c_5,&
      gpufortrt_use_device_c_6,&
      gpufortrt_use_device_c_7,&
      gpufortrt_use_device_i2_0,&
      gpufortrt_use_device_i2_1,&
      gpufortrt_use_device_i2_2,&
      gpufortrt_use_device_i2_3,&
      gpufortrt_use_device_i2_4,&
      gpufortrt_use_device_i2_5,&
      gpufortrt_use_device_i2_6,&
      gpufortrt_use_device_i2_7,&
      gpufortrt_use_device_i4_0,&
      gpufortrt_use_device_i4_1,&
      gpufortrt_use_device_i4_2,&
      gpufortrt_use_device_i4_3,&
      gpufortrt_use_device_i4_4,&
      gpufortrt_use_device_i4_5,&
      gpufortrt_use_device_i4_6,&
      gpufortrt_use_device_i4_7,&
      gpufortrt_use_device_i8_0,&
      gpufortrt_use_device_i8_1,&
      gpufortrt_use_device_i8_2,&
      gpufortrt_use_device_i8_3,&
      gpufortrt_use_device_i8_4,&
      gpufortrt_use_device_i8_5,&
      gpufortrt_use_device_i8_6,&
      gpufortrt_use_device_i8_7,&
      gpufortrt_use_device_r4_0,&
      gpufortrt_use_device_r4_1,&
      gpufortrt_use_device_r4_2,&
      gpufortrt_use_device_r4_3,&
      gpufortrt_use_device_r4_4,&
      gpufortrt_use_device_r4_5,&
      gpufortrt_use_device_r4_6,&
      gpufortrt_use_device_r4_7,&
      gpufortrt_use_device_r8_0,&
      gpufortrt_use_device_r8_1,&
      gpufortrt_use_device_r8_2,&
      gpufortrt_use_device_r8_3,&
      gpufortrt_use_device_r8_4,&
      gpufortrt_use_device_r8_5,&
      gpufortrt_use_device_r8_6,&
      gpufortrt_use_device_r8_7,&
      gpufortrt_use_device_c4_0,&
      gpufortrt_use_device_c4_1,&
      gpufortrt_use_device_c4_2,&
      gpufortrt_use_device_c4_3,&
      gpufortrt_use_device_c4_4,&
      gpufortrt_use_device_c4_5,&
      gpufortrt_use_device_c4_6,&
      gpufortrt_use_device_c4_7,&
      gpufortrt_use_device_c8_0,&
      gpufortrt_use_device_c8_1,&
      gpufortrt_use_device_c8_2,&
      gpufortrt_use_device_c8_3,&
      gpufortrt_use_device_c8_4,&
      gpufortrt_use_device_c8_5,&
      gpufortrt_use_device_c8_6,&
      gpufortrt_use_device_c8_7
  end interface


  !> Decrement the structured reference counter of a record.
  interface gpufortrt_map_dec_struct_refs
module procedure :: gpufortrt_map_dec_struct_refs_l_scal,&
                        gpufortrt_map_dec_struct_refs_l_arr
module procedure :: gpufortrt_map_dec_struct_refs_c_scal,&
                        gpufortrt_map_dec_struct_refs_c_arr
module procedure :: gpufortrt_map_dec_struct_refs_i2_scal,&
                        gpufortrt_map_dec_struct_refs_i2_arr
module procedure :: gpufortrt_map_dec_struct_refs_i4_scal,&
                        gpufortrt_map_dec_struct_refs_i4_arr
module procedure :: gpufortrt_map_dec_struct_refs_i8_scal,&
                        gpufortrt_map_dec_struct_refs_i8_arr
module procedure :: gpufortrt_map_dec_struct_refs_r4_scal,&
                        gpufortrt_map_dec_struct_refs_r4_arr
module procedure :: gpufortrt_map_dec_struct_refs_r8_scal,&
                        gpufortrt_map_dec_struct_refs_r8_arr
module procedure :: gpufortrt_map_dec_struct_refs_c4_scal,&
                        gpufortrt_map_dec_struct_refs_c4_arr
module procedure :: gpufortrt_map_dec_struct_refs_c8_scal,&
                        gpufortrt_map_dec_struct_refs_c8_arr
  end interface

 
  interface gpufortrt_delete
    module procedure &
      gpufortrt_delete_b, &
      gpufortrt_delete_l_0,&
      gpufortrt_delete_l_1,&
      gpufortrt_delete_l_2,&
      gpufortrt_delete_l_3,&
      gpufortrt_delete_l_4,&
      gpufortrt_delete_l_5,&
      gpufortrt_delete_l_6,&
      gpufortrt_delete_l_7,&
      gpufortrt_delete_c_0,&
      gpufortrt_delete_c_1,&
      gpufortrt_delete_c_2,&
      gpufortrt_delete_c_3,&
      gpufortrt_delete_c_4,&
      gpufortrt_delete_c_5,&
      gpufortrt_delete_c_6,&
      gpufortrt_delete_c_7,&
      gpufortrt_delete_i2_0,&
      gpufortrt_delete_i2_1,&
      gpufortrt_delete_i2_2,&
      gpufortrt_delete_i2_3,&
      gpufortrt_delete_i2_4,&
      gpufortrt_delete_i2_5,&
      gpufortrt_delete_i2_6,&
      gpufortrt_delete_i2_7,&
      gpufortrt_delete_i4_0,&
      gpufortrt_delete_i4_1,&
      gpufortrt_delete_i4_2,&
      gpufortrt_delete_i4_3,&
      gpufortrt_delete_i4_4,&
      gpufortrt_delete_i4_5,&
      gpufortrt_delete_i4_6,&
      gpufortrt_delete_i4_7,&
      gpufortrt_delete_i8_0,&
      gpufortrt_delete_i8_1,&
      gpufortrt_delete_i8_2,&
      gpufortrt_delete_i8_3,&
      gpufortrt_delete_i8_4,&
      gpufortrt_delete_i8_5,&
      gpufortrt_delete_i8_6,&
      gpufortrt_delete_i8_7,&
      gpufortrt_delete_r4_0,&
      gpufortrt_delete_r4_1,&
      gpufortrt_delete_r4_2,&
      gpufortrt_delete_r4_3,&
      gpufortrt_delete_r4_4,&
      gpufortrt_delete_r4_5,&
      gpufortrt_delete_r4_6,&
      gpufortrt_delete_r4_7,&
      gpufortrt_delete_r8_0,&
      gpufortrt_delete_r8_1,&
      gpufortrt_delete_r8_2,&
      gpufortrt_delete_r8_3,&
      gpufortrt_delete_r8_4,&
      gpufortrt_delete_r8_5,&
      gpufortrt_delete_r8_6,&
      gpufortrt_delete_r8_7,&
      gpufortrt_delete_c4_0,&
      gpufortrt_delete_c4_1,&
      gpufortrt_delete_c4_2,&
      gpufortrt_delete_c4_3,&
      gpufortrt_delete_c4_4,&
      gpufortrt_delete_c4_5,&
      gpufortrt_delete_c4_6,&
      gpufortrt_delete_c4_7,&
      gpufortrt_delete_c8_0,&
      gpufortrt_delete_c8_1,&
      gpufortrt_delete_c8_2,&
      gpufortrt_delete_c8_3,&
      gpufortrt_delete_c8_4,&
      gpufortrt_delete_c8_5,&
      gpufortrt_delete_c8_6,&
      gpufortrt_delete_c8_7
  end interface


 
  interface gpufortrt_present
    module procedure &
      gpufortrt_present_b, &
      gpufortrt_present_l_0,&
      gpufortrt_present_l_1,&
      gpufortrt_present_l_2,&
      gpufortrt_present_l_3,&
      gpufortrt_present_l_4,&
      gpufortrt_present_l_5,&
      gpufortrt_present_l_6,&
      gpufortrt_present_l_7,&
      gpufortrt_present_c_0,&
      gpufortrt_present_c_1,&
      gpufortrt_present_c_2,&
      gpufortrt_present_c_3,&
      gpufortrt_present_c_4,&
      gpufortrt_present_c_5,&
      gpufortrt_present_c_6,&
      gpufortrt_present_c_7,&
      gpufortrt_present_i2_0,&
      gpufortrt_present_i2_1,&
      gpufortrt_present_i2_2,&
      gpufortrt_present_i2_3,&
      gpufortrt_present_i2_4,&
      gpufortrt_present_i2_5,&
      gpufortrt_present_i2_6,&
      gpufortrt_present_i2_7,&
      gpufortrt_present_i4_0,&
      gpufortrt_present_i4_1,&
      gpufortrt_present_i4_2,&
      gpufortrt_present_i4_3,&
      gpufortrt_present_i4_4,&
      gpufortrt_present_i4_5,&
      gpufortrt_present_i4_6,&
      gpufortrt_present_i4_7,&
      gpufortrt_present_i8_0,&
      gpufortrt_present_i8_1,&
      gpufortrt_present_i8_2,&
      gpufortrt_present_i8_3,&
      gpufortrt_present_i8_4,&
      gpufortrt_present_i8_5,&
      gpufortrt_present_i8_6,&
      gpufortrt_present_i8_7,&
      gpufortrt_present_r4_0,&
      gpufortrt_present_r4_1,&
      gpufortrt_present_r4_2,&
      gpufortrt_present_r4_3,&
      gpufortrt_present_r4_4,&
      gpufortrt_present_r4_5,&
      gpufortrt_present_r4_6,&
      gpufortrt_present_r4_7,&
      gpufortrt_present_r8_0,&
      gpufortrt_present_r8_1,&
      gpufortrt_present_r8_2,&
      gpufortrt_present_r8_3,&
      gpufortrt_present_r8_4,&
      gpufortrt_present_r8_5,&
      gpufortrt_present_r8_6,&
      gpufortrt_present_r8_7,&
      gpufortrt_present_c4_0,&
      gpufortrt_present_c4_1,&
      gpufortrt_present_c4_2,&
      gpufortrt_present_c4_3,&
      gpufortrt_present_c4_4,&
      gpufortrt_present_c4_5,&
      gpufortrt_present_c4_6,&
      gpufortrt_present_c4_7,&
      gpufortrt_present_c8_0,&
      gpufortrt_present_c8_1,&
      gpufortrt_present_c8_2,&
      gpufortrt_present_c8_3,&
      gpufortrt_present_c8_4,&
      gpufortrt_present_c8_5,&
      gpufortrt_present_c8_6,&
      gpufortrt_present_c8_7
  end interface


 
  interface gpufortrt_no_create
    module procedure &
      gpufortrt_no_create_b, &
      gpufortrt_no_create_l_0,&
      gpufortrt_no_create_l_1,&
      gpufortrt_no_create_l_2,&
      gpufortrt_no_create_l_3,&
      gpufortrt_no_create_l_4,&
      gpufortrt_no_create_l_5,&
      gpufortrt_no_create_l_6,&
      gpufortrt_no_create_l_7,&
      gpufortrt_no_create_c_0,&
      gpufortrt_no_create_c_1,&
      gpufortrt_no_create_c_2,&
      gpufortrt_no_create_c_3,&
      gpufortrt_no_create_c_4,&
      gpufortrt_no_create_c_5,&
      gpufortrt_no_create_c_6,&
      gpufortrt_no_create_c_7,&
      gpufortrt_no_create_i2_0,&
      gpufortrt_no_create_i2_1,&
      gpufortrt_no_create_i2_2,&
      gpufortrt_no_create_i2_3,&
      gpufortrt_no_create_i2_4,&
      gpufortrt_no_create_i2_5,&
      gpufortrt_no_create_i2_6,&
      gpufortrt_no_create_i2_7,&
      gpufortrt_no_create_i4_0,&
      gpufortrt_no_create_i4_1,&
      gpufortrt_no_create_i4_2,&
      gpufortrt_no_create_i4_3,&
      gpufortrt_no_create_i4_4,&
      gpufortrt_no_create_i4_5,&
      gpufortrt_no_create_i4_6,&
      gpufortrt_no_create_i4_7,&
      gpufortrt_no_create_i8_0,&
      gpufortrt_no_create_i8_1,&
      gpufortrt_no_create_i8_2,&
      gpufortrt_no_create_i8_3,&
      gpufortrt_no_create_i8_4,&
      gpufortrt_no_create_i8_5,&
      gpufortrt_no_create_i8_6,&
      gpufortrt_no_create_i8_7,&
      gpufortrt_no_create_r4_0,&
      gpufortrt_no_create_r4_1,&
      gpufortrt_no_create_r4_2,&
      gpufortrt_no_create_r4_3,&
      gpufortrt_no_create_r4_4,&
      gpufortrt_no_create_r4_5,&
      gpufortrt_no_create_r4_6,&
      gpufortrt_no_create_r4_7,&
      gpufortrt_no_create_r8_0,&
      gpufortrt_no_create_r8_1,&
      gpufortrt_no_create_r8_2,&
      gpufortrt_no_create_r8_3,&
      gpufortrt_no_create_r8_4,&
      gpufortrt_no_create_r8_5,&
      gpufortrt_no_create_r8_6,&
      gpufortrt_no_create_r8_7,&
      gpufortrt_no_create_c4_0,&
      gpufortrt_no_create_c4_1,&
      gpufortrt_no_create_c4_2,&
      gpufortrt_no_create_c4_3,&
      gpufortrt_no_create_c4_4,&
      gpufortrt_no_create_c4_5,&
      gpufortrt_no_create_c4_6,&
      gpufortrt_no_create_c4_7,&
      gpufortrt_no_create_c8_0,&
      gpufortrt_no_create_c8_1,&
      gpufortrt_no_create_c8_2,&
      gpufortrt_no_create_c8_3,&
      gpufortrt_no_create_c8_4,&
      gpufortrt_no_create_c8_5,&
      gpufortrt_no_create_c8_6,&
      gpufortrt_no_create_c8_7
  end interface


 
  interface gpufortrt_create
    module procedure &
      gpufortrt_create_b, &
      gpufortrt_create_l_0,&
      gpufortrt_create_l_1,&
      gpufortrt_create_l_2,&
      gpufortrt_create_l_3,&
      gpufortrt_create_l_4,&
      gpufortrt_create_l_5,&
      gpufortrt_create_l_6,&
      gpufortrt_create_l_7,&
      gpufortrt_create_c_0,&
      gpufortrt_create_c_1,&
      gpufortrt_create_c_2,&
      gpufortrt_create_c_3,&
      gpufortrt_create_c_4,&
      gpufortrt_create_c_5,&
      gpufortrt_create_c_6,&
      gpufortrt_create_c_7,&
      gpufortrt_create_i2_0,&
      gpufortrt_create_i2_1,&
      gpufortrt_create_i2_2,&
      gpufortrt_create_i2_3,&
      gpufortrt_create_i2_4,&
      gpufortrt_create_i2_5,&
      gpufortrt_create_i2_6,&
      gpufortrt_create_i2_7,&
      gpufortrt_create_i4_0,&
      gpufortrt_create_i4_1,&
      gpufortrt_create_i4_2,&
      gpufortrt_create_i4_3,&
      gpufortrt_create_i4_4,&
      gpufortrt_create_i4_5,&
      gpufortrt_create_i4_6,&
      gpufortrt_create_i4_7,&
      gpufortrt_create_i8_0,&
      gpufortrt_create_i8_1,&
      gpufortrt_create_i8_2,&
      gpufortrt_create_i8_3,&
      gpufortrt_create_i8_4,&
      gpufortrt_create_i8_5,&
      gpufortrt_create_i8_6,&
      gpufortrt_create_i8_7,&
      gpufortrt_create_r4_0,&
      gpufortrt_create_r4_1,&
      gpufortrt_create_r4_2,&
      gpufortrt_create_r4_3,&
      gpufortrt_create_r4_4,&
      gpufortrt_create_r4_5,&
      gpufortrt_create_r4_6,&
      gpufortrt_create_r4_7,&
      gpufortrt_create_r8_0,&
      gpufortrt_create_r8_1,&
      gpufortrt_create_r8_2,&
      gpufortrt_create_r8_3,&
      gpufortrt_create_r8_4,&
      gpufortrt_create_r8_5,&
      gpufortrt_create_r8_6,&
      gpufortrt_create_r8_7,&
      gpufortrt_create_c4_0,&
      gpufortrt_create_c4_1,&
      gpufortrt_create_c4_2,&
      gpufortrt_create_c4_3,&
      gpufortrt_create_c4_4,&
      gpufortrt_create_c4_5,&
      gpufortrt_create_c4_6,&
      gpufortrt_create_c4_7,&
      gpufortrt_create_c8_0,&
      gpufortrt_create_c8_1,&
      gpufortrt_create_c8_2,&
      gpufortrt_create_c8_3,&
      gpufortrt_create_c8_4,&
      gpufortrt_create_c8_5,&
      gpufortrt_create_c8_6,&
      gpufortrt_create_c8_7
  end interface


 
  interface gpufortrt_copy
    module procedure &
      gpufortrt_copy_b, &
      gpufortrt_copy_l_0,&
      gpufortrt_copy_l_1,&
      gpufortrt_copy_l_2,&
      gpufortrt_copy_l_3,&
      gpufortrt_copy_l_4,&
      gpufortrt_copy_l_5,&
      gpufortrt_copy_l_6,&
      gpufortrt_copy_l_7,&
      gpufortrt_copy_c_0,&
      gpufortrt_copy_c_1,&
      gpufortrt_copy_c_2,&
      gpufortrt_copy_c_3,&
      gpufortrt_copy_c_4,&
      gpufortrt_copy_c_5,&
      gpufortrt_copy_c_6,&
      gpufortrt_copy_c_7,&
      gpufortrt_copy_i2_0,&
      gpufortrt_copy_i2_1,&
      gpufortrt_copy_i2_2,&
      gpufortrt_copy_i2_3,&
      gpufortrt_copy_i2_4,&
      gpufortrt_copy_i2_5,&
      gpufortrt_copy_i2_6,&
      gpufortrt_copy_i2_7,&
      gpufortrt_copy_i4_0,&
      gpufortrt_copy_i4_1,&
      gpufortrt_copy_i4_2,&
      gpufortrt_copy_i4_3,&
      gpufortrt_copy_i4_4,&
      gpufortrt_copy_i4_5,&
      gpufortrt_copy_i4_6,&
      gpufortrt_copy_i4_7,&
      gpufortrt_copy_i8_0,&
      gpufortrt_copy_i8_1,&
      gpufortrt_copy_i8_2,&
      gpufortrt_copy_i8_3,&
      gpufortrt_copy_i8_4,&
      gpufortrt_copy_i8_5,&
      gpufortrt_copy_i8_6,&
      gpufortrt_copy_i8_7,&
      gpufortrt_copy_r4_0,&
      gpufortrt_copy_r4_1,&
      gpufortrt_copy_r4_2,&
      gpufortrt_copy_r4_3,&
      gpufortrt_copy_r4_4,&
      gpufortrt_copy_r4_5,&
      gpufortrt_copy_r4_6,&
      gpufortrt_copy_r4_7,&
      gpufortrt_copy_r8_0,&
      gpufortrt_copy_r8_1,&
      gpufortrt_copy_r8_2,&
      gpufortrt_copy_r8_3,&
      gpufortrt_copy_r8_4,&
      gpufortrt_copy_r8_5,&
      gpufortrt_copy_r8_6,&
      gpufortrt_copy_r8_7,&
      gpufortrt_copy_c4_0,&
      gpufortrt_copy_c4_1,&
      gpufortrt_copy_c4_2,&
      gpufortrt_copy_c4_3,&
      gpufortrt_copy_c4_4,&
      gpufortrt_copy_c4_5,&
      gpufortrt_copy_c4_6,&
      gpufortrt_copy_c4_7,&
      gpufortrt_copy_c8_0,&
      gpufortrt_copy_c8_1,&
      gpufortrt_copy_c8_2,&
      gpufortrt_copy_c8_3,&
      gpufortrt_copy_c8_4,&
      gpufortrt_copy_c8_5,&
      gpufortrt_copy_c8_6,&
      gpufortrt_copy_c8_7
  end interface


 
  interface gpufortrt_copyin
    module procedure &
      gpufortrt_copyin_b, &
      gpufortrt_copyin_l_0,&
      gpufortrt_copyin_l_1,&
      gpufortrt_copyin_l_2,&
      gpufortrt_copyin_l_3,&
      gpufortrt_copyin_l_4,&
      gpufortrt_copyin_l_5,&
      gpufortrt_copyin_l_6,&
      gpufortrt_copyin_l_7,&
      gpufortrt_copyin_c_0,&
      gpufortrt_copyin_c_1,&
      gpufortrt_copyin_c_2,&
      gpufortrt_copyin_c_3,&
      gpufortrt_copyin_c_4,&
      gpufortrt_copyin_c_5,&
      gpufortrt_copyin_c_6,&
      gpufortrt_copyin_c_7,&
      gpufortrt_copyin_i2_0,&
      gpufortrt_copyin_i2_1,&
      gpufortrt_copyin_i2_2,&
      gpufortrt_copyin_i2_3,&
      gpufortrt_copyin_i2_4,&
      gpufortrt_copyin_i2_5,&
      gpufortrt_copyin_i2_6,&
      gpufortrt_copyin_i2_7,&
      gpufortrt_copyin_i4_0,&
      gpufortrt_copyin_i4_1,&
      gpufortrt_copyin_i4_2,&
      gpufortrt_copyin_i4_3,&
      gpufortrt_copyin_i4_4,&
      gpufortrt_copyin_i4_5,&
      gpufortrt_copyin_i4_6,&
      gpufortrt_copyin_i4_7,&
      gpufortrt_copyin_i8_0,&
      gpufortrt_copyin_i8_1,&
      gpufortrt_copyin_i8_2,&
      gpufortrt_copyin_i8_3,&
      gpufortrt_copyin_i8_4,&
      gpufortrt_copyin_i8_5,&
      gpufortrt_copyin_i8_6,&
      gpufortrt_copyin_i8_7,&
      gpufortrt_copyin_r4_0,&
      gpufortrt_copyin_r4_1,&
      gpufortrt_copyin_r4_2,&
      gpufortrt_copyin_r4_3,&
      gpufortrt_copyin_r4_4,&
      gpufortrt_copyin_r4_5,&
      gpufortrt_copyin_r4_6,&
      gpufortrt_copyin_r4_7,&
      gpufortrt_copyin_r8_0,&
      gpufortrt_copyin_r8_1,&
      gpufortrt_copyin_r8_2,&
      gpufortrt_copyin_r8_3,&
      gpufortrt_copyin_r8_4,&
      gpufortrt_copyin_r8_5,&
      gpufortrt_copyin_r8_6,&
      gpufortrt_copyin_r8_7,&
      gpufortrt_copyin_c4_0,&
      gpufortrt_copyin_c4_1,&
      gpufortrt_copyin_c4_2,&
      gpufortrt_copyin_c4_3,&
      gpufortrt_copyin_c4_4,&
      gpufortrt_copyin_c4_5,&
      gpufortrt_copyin_c4_6,&
      gpufortrt_copyin_c4_7,&
      gpufortrt_copyin_c8_0,&
      gpufortrt_copyin_c8_1,&
      gpufortrt_copyin_c8_2,&
      gpufortrt_copyin_c8_3,&
      gpufortrt_copyin_c8_4,&
      gpufortrt_copyin_c8_5,&
      gpufortrt_copyin_c8_6,&
      gpufortrt_copyin_c8_7
  end interface


 
  interface gpufortrt_copyout
    module procedure &
      gpufortrt_copyout_b, &
      gpufortrt_copyout_l_0,&
      gpufortrt_copyout_l_1,&
      gpufortrt_copyout_l_2,&
      gpufortrt_copyout_l_3,&
      gpufortrt_copyout_l_4,&
      gpufortrt_copyout_l_5,&
      gpufortrt_copyout_l_6,&
      gpufortrt_copyout_l_7,&
      gpufortrt_copyout_c_0,&
      gpufortrt_copyout_c_1,&
      gpufortrt_copyout_c_2,&
      gpufortrt_copyout_c_3,&
      gpufortrt_copyout_c_4,&
      gpufortrt_copyout_c_5,&
      gpufortrt_copyout_c_6,&
      gpufortrt_copyout_c_7,&
      gpufortrt_copyout_i2_0,&
      gpufortrt_copyout_i2_1,&
      gpufortrt_copyout_i2_2,&
      gpufortrt_copyout_i2_3,&
      gpufortrt_copyout_i2_4,&
      gpufortrt_copyout_i2_5,&
      gpufortrt_copyout_i2_6,&
      gpufortrt_copyout_i2_7,&
      gpufortrt_copyout_i4_0,&
      gpufortrt_copyout_i4_1,&
      gpufortrt_copyout_i4_2,&
      gpufortrt_copyout_i4_3,&
      gpufortrt_copyout_i4_4,&
      gpufortrt_copyout_i4_5,&
      gpufortrt_copyout_i4_6,&
      gpufortrt_copyout_i4_7,&
      gpufortrt_copyout_i8_0,&
      gpufortrt_copyout_i8_1,&
      gpufortrt_copyout_i8_2,&
      gpufortrt_copyout_i8_3,&
      gpufortrt_copyout_i8_4,&
      gpufortrt_copyout_i8_5,&
      gpufortrt_copyout_i8_6,&
      gpufortrt_copyout_i8_7,&
      gpufortrt_copyout_r4_0,&
      gpufortrt_copyout_r4_1,&
      gpufortrt_copyout_r4_2,&
      gpufortrt_copyout_r4_3,&
      gpufortrt_copyout_r4_4,&
      gpufortrt_copyout_r4_5,&
      gpufortrt_copyout_r4_6,&
      gpufortrt_copyout_r4_7,&
      gpufortrt_copyout_r8_0,&
      gpufortrt_copyout_r8_1,&
      gpufortrt_copyout_r8_2,&
      gpufortrt_copyout_r8_3,&
      gpufortrt_copyout_r8_4,&
      gpufortrt_copyout_r8_5,&
      gpufortrt_copyout_r8_6,&
      gpufortrt_copyout_r8_7,&
      gpufortrt_copyout_c4_0,&
      gpufortrt_copyout_c4_1,&
      gpufortrt_copyout_c4_2,&
      gpufortrt_copyout_c4_3,&
      gpufortrt_copyout_c4_4,&
      gpufortrt_copyout_c4_5,&
      gpufortrt_copyout_c4_6,&
      gpufortrt_copyout_c4_7,&
      gpufortrt_copyout_c8_0,&
      gpufortrt_copyout_c8_1,&
      gpufortrt_copyout_c8_2,&
      gpufortrt_copyout_c8_3,&
      gpufortrt_copyout_c8_4,&
      gpufortrt_copyout_c8_5,&
      gpufortrt_copyout_c8_6,&
      gpufortrt_copyout_c8_7
  end interface


 
  interface gpufortrt_map_delete
    module procedure &
      gpufortrt_map_delete_b, &
      gpufortrt_map_delete_l_0,&
      gpufortrt_map_delete_l_1,&
      gpufortrt_map_delete_l_2,&
      gpufortrt_map_delete_l_3,&
      gpufortrt_map_delete_l_4,&
      gpufortrt_map_delete_l_5,&
      gpufortrt_map_delete_l_6,&
      gpufortrt_map_delete_l_7,&
      gpufortrt_map_delete_c_0,&
      gpufortrt_map_delete_c_1,&
      gpufortrt_map_delete_c_2,&
      gpufortrt_map_delete_c_3,&
      gpufortrt_map_delete_c_4,&
      gpufortrt_map_delete_c_5,&
      gpufortrt_map_delete_c_6,&
      gpufortrt_map_delete_c_7,&
      gpufortrt_map_delete_i2_0,&
      gpufortrt_map_delete_i2_1,&
      gpufortrt_map_delete_i2_2,&
      gpufortrt_map_delete_i2_3,&
      gpufortrt_map_delete_i2_4,&
      gpufortrt_map_delete_i2_5,&
      gpufortrt_map_delete_i2_6,&
      gpufortrt_map_delete_i2_7,&
      gpufortrt_map_delete_i4_0,&
      gpufortrt_map_delete_i4_1,&
      gpufortrt_map_delete_i4_2,&
      gpufortrt_map_delete_i4_3,&
      gpufortrt_map_delete_i4_4,&
      gpufortrt_map_delete_i4_5,&
      gpufortrt_map_delete_i4_6,&
      gpufortrt_map_delete_i4_7,&
      gpufortrt_map_delete_i8_0,&
      gpufortrt_map_delete_i8_1,&
      gpufortrt_map_delete_i8_2,&
      gpufortrt_map_delete_i8_3,&
      gpufortrt_map_delete_i8_4,&
      gpufortrt_map_delete_i8_5,&
      gpufortrt_map_delete_i8_6,&
      gpufortrt_map_delete_i8_7,&
      gpufortrt_map_delete_r4_0,&
      gpufortrt_map_delete_r4_1,&
      gpufortrt_map_delete_r4_2,&
      gpufortrt_map_delete_r4_3,&
      gpufortrt_map_delete_r4_4,&
      gpufortrt_map_delete_r4_5,&
      gpufortrt_map_delete_r4_6,&
      gpufortrt_map_delete_r4_7,&
      gpufortrt_map_delete_r8_0,&
      gpufortrt_map_delete_r8_1,&
      gpufortrt_map_delete_r8_2,&
      gpufortrt_map_delete_r8_3,&
      gpufortrt_map_delete_r8_4,&
      gpufortrt_map_delete_r8_5,&
      gpufortrt_map_delete_r8_6,&
      gpufortrt_map_delete_r8_7,&
      gpufortrt_map_delete_c4_0,&
      gpufortrt_map_delete_c4_1,&
      gpufortrt_map_delete_c4_2,&
      gpufortrt_map_delete_c4_3,&
      gpufortrt_map_delete_c4_4,&
      gpufortrt_map_delete_c4_5,&
      gpufortrt_map_delete_c4_6,&
      gpufortrt_map_delete_c4_7,&
      gpufortrt_map_delete_c8_0,&
      gpufortrt_map_delete_c8_1,&
      gpufortrt_map_delete_c8_2,&
      gpufortrt_map_delete_c8_3,&
      gpufortrt_map_delete_c8_4,&
      gpufortrt_map_delete_c8_5,&
      gpufortrt_map_delete_c8_6,&
      gpufortrt_map_delete_c8_7
  end interface


 
  interface gpufortrt_map_present
    module procedure &
      gpufortrt_map_present_b, &
      gpufortrt_map_present_l_0,&
      gpufortrt_map_present_l_1,&
      gpufortrt_map_present_l_2,&
      gpufortrt_map_present_l_3,&
      gpufortrt_map_present_l_4,&
      gpufortrt_map_present_l_5,&
      gpufortrt_map_present_l_6,&
      gpufortrt_map_present_l_7,&
      gpufortrt_map_present_c_0,&
      gpufortrt_map_present_c_1,&
      gpufortrt_map_present_c_2,&
      gpufortrt_map_present_c_3,&
      gpufortrt_map_present_c_4,&
      gpufortrt_map_present_c_5,&
      gpufortrt_map_present_c_6,&
      gpufortrt_map_present_c_7,&
      gpufortrt_map_present_i2_0,&
      gpufortrt_map_present_i2_1,&
      gpufortrt_map_present_i2_2,&
      gpufortrt_map_present_i2_3,&
      gpufortrt_map_present_i2_4,&
      gpufortrt_map_present_i2_5,&
      gpufortrt_map_present_i2_6,&
      gpufortrt_map_present_i2_7,&
      gpufortrt_map_present_i4_0,&
      gpufortrt_map_present_i4_1,&
      gpufortrt_map_present_i4_2,&
      gpufortrt_map_present_i4_3,&
      gpufortrt_map_present_i4_4,&
      gpufortrt_map_present_i4_5,&
      gpufortrt_map_present_i4_6,&
      gpufortrt_map_present_i4_7,&
      gpufortrt_map_present_i8_0,&
      gpufortrt_map_present_i8_1,&
      gpufortrt_map_present_i8_2,&
      gpufortrt_map_present_i8_3,&
      gpufortrt_map_present_i8_4,&
      gpufortrt_map_present_i8_5,&
      gpufortrt_map_present_i8_6,&
      gpufortrt_map_present_i8_7,&
      gpufortrt_map_present_r4_0,&
      gpufortrt_map_present_r4_1,&
      gpufortrt_map_present_r4_2,&
      gpufortrt_map_present_r4_3,&
      gpufortrt_map_present_r4_4,&
      gpufortrt_map_present_r4_5,&
      gpufortrt_map_present_r4_6,&
      gpufortrt_map_present_r4_7,&
      gpufortrt_map_present_r8_0,&
      gpufortrt_map_present_r8_1,&
      gpufortrt_map_present_r8_2,&
      gpufortrt_map_present_r8_3,&
      gpufortrt_map_present_r8_4,&
      gpufortrt_map_present_r8_5,&
      gpufortrt_map_present_r8_6,&
      gpufortrt_map_present_r8_7,&
      gpufortrt_map_present_c4_0,&
      gpufortrt_map_present_c4_1,&
      gpufortrt_map_present_c4_2,&
      gpufortrt_map_present_c4_3,&
      gpufortrt_map_present_c4_4,&
      gpufortrt_map_present_c4_5,&
      gpufortrt_map_present_c4_6,&
      gpufortrt_map_present_c4_7,&
      gpufortrt_map_present_c8_0,&
      gpufortrt_map_present_c8_1,&
      gpufortrt_map_present_c8_2,&
      gpufortrt_map_present_c8_3,&
      gpufortrt_map_present_c8_4,&
      gpufortrt_map_present_c8_5,&
      gpufortrt_map_present_c8_6,&
      gpufortrt_map_present_c8_7
  end interface


 
  interface gpufortrt_map_no_create
    module procedure &
      gpufortrt_map_no_create_b, &
      gpufortrt_map_no_create_l_0,&
      gpufortrt_map_no_create_l_1,&
      gpufortrt_map_no_create_l_2,&
      gpufortrt_map_no_create_l_3,&
      gpufortrt_map_no_create_l_4,&
      gpufortrt_map_no_create_l_5,&
      gpufortrt_map_no_create_l_6,&
      gpufortrt_map_no_create_l_7,&
      gpufortrt_map_no_create_c_0,&
      gpufortrt_map_no_create_c_1,&
      gpufortrt_map_no_create_c_2,&
      gpufortrt_map_no_create_c_3,&
      gpufortrt_map_no_create_c_4,&
      gpufortrt_map_no_create_c_5,&
      gpufortrt_map_no_create_c_6,&
      gpufortrt_map_no_create_c_7,&
      gpufortrt_map_no_create_i2_0,&
      gpufortrt_map_no_create_i2_1,&
      gpufortrt_map_no_create_i2_2,&
      gpufortrt_map_no_create_i2_3,&
      gpufortrt_map_no_create_i2_4,&
      gpufortrt_map_no_create_i2_5,&
      gpufortrt_map_no_create_i2_6,&
      gpufortrt_map_no_create_i2_7,&
      gpufortrt_map_no_create_i4_0,&
      gpufortrt_map_no_create_i4_1,&
      gpufortrt_map_no_create_i4_2,&
      gpufortrt_map_no_create_i4_3,&
      gpufortrt_map_no_create_i4_4,&
      gpufortrt_map_no_create_i4_5,&
      gpufortrt_map_no_create_i4_6,&
      gpufortrt_map_no_create_i4_7,&
      gpufortrt_map_no_create_i8_0,&
      gpufortrt_map_no_create_i8_1,&
      gpufortrt_map_no_create_i8_2,&
      gpufortrt_map_no_create_i8_3,&
      gpufortrt_map_no_create_i8_4,&
      gpufortrt_map_no_create_i8_5,&
      gpufortrt_map_no_create_i8_6,&
      gpufortrt_map_no_create_i8_7,&
      gpufortrt_map_no_create_r4_0,&
      gpufortrt_map_no_create_r4_1,&
      gpufortrt_map_no_create_r4_2,&
      gpufortrt_map_no_create_r4_3,&
      gpufortrt_map_no_create_r4_4,&
      gpufortrt_map_no_create_r4_5,&
      gpufortrt_map_no_create_r4_6,&
      gpufortrt_map_no_create_r4_7,&
      gpufortrt_map_no_create_r8_0,&
      gpufortrt_map_no_create_r8_1,&
      gpufortrt_map_no_create_r8_2,&
      gpufortrt_map_no_create_r8_3,&
      gpufortrt_map_no_create_r8_4,&
      gpufortrt_map_no_create_r8_5,&
      gpufortrt_map_no_create_r8_6,&
      gpufortrt_map_no_create_r8_7,&
      gpufortrt_map_no_create_c4_0,&
      gpufortrt_map_no_create_c4_1,&
      gpufortrt_map_no_create_c4_2,&
      gpufortrt_map_no_create_c4_3,&
      gpufortrt_map_no_create_c4_4,&
      gpufortrt_map_no_create_c4_5,&
      gpufortrt_map_no_create_c4_6,&
      gpufortrt_map_no_create_c4_7,&
      gpufortrt_map_no_create_c8_0,&
      gpufortrt_map_no_create_c8_1,&
      gpufortrt_map_no_create_c8_2,&
      gpufortrt_map_no_create_c8_3,&
      gpufortrt_map_no_create_c8_4,&
      gpufortrt_map_no_create_c8_5,&
      gpufortrt_map_no_create_c8_6,&
      gpufortrt_map_no_create_c8_7
  end interface


 
  interface gpufortrt_map_create
    module procedure &
      gpufortrt_map_create_b, &
      gpufortrt_map_create_l_0,&
      gpufortrt_map_create_l_1,&
      gpufortrt_map_create_l_2,&
      gpufortrt_map_create_l_3,&
      gpufortrt_map_create_l_4,&
      gpufortrt_map_create_l_5,&
      gpufortrt_map_create_l_6,&
      gpufortrt_map_create_l_7,&
      gpufortrt_map_create_c_0,&
      gpufortrt_map_create_c_1,&
      gpufortrt_map_create_c_2,&
      gpufortrt_map_create_c_3,&
      gpufortrt_map_create_c_4,&
      gpufortrt_map_create_c_5,&
      gpufortrt_map_create_c_6,&
      gpufortrt_map_create_c_7,&
      gpufortrt_map_create_i2_0,&
      gpufortrt_map_create_i2_1,&
      gpufortrt_map_create_i2_2,&
      gpufortrt_map_create_i2_3,&
      gpufortrt_map_create_i2_4,&
      gpufortrt_map_create_i2_5,&
      gpufortrt_map_create_i2_6,&
      gpufortrt_map_create_i2_7,&
      gpufortrt_map_create_i4_0,&
      gpufortrt_map_create_i4_1,&
      gpufortrt_map_create_i4_2,&
      gpufortrt_map_create_i4_3,&
      gpufortrt_map_create_i4_4,&
      gpufortrt_map_create_i4_5,&
      gpufortrt_map_create_i4_6,&
      gpufortrt_map_create_i4_7,&
      gpufortrt_map_create_i8_0,&
      gpufortrt_map_create_i8_1,&
      gpufortrt_map_create_i8_2,&
      gpufortrt_map_create_i8_3,&
      gpufortrt_map_create_i8_4,&
      gpufortrt_map_create_i8_5,&
      gpufortrt_map_create_i8_6,&
      gpufortrt_map_create_i8_7,&
      gpufortrt_map_create_r4_0,&
      gpufortrt_map_create_r4_1,&
      gpufortrt_map_create_r4_2,&
      gpufortrt_map_create_r4_3,&
      gpufortrt_map_create_r4_4,&
      gpufortrt_map_create_r4_5,&
      gpufortrt_map_create_r4_6,&
      gpufortrt_map_create_r4_7,&
      gpufortrt_map_create_r8_0,&
      gpufortrt_map_create_r8_1,&
      gpufortrt_map_create_r8_2,&
      gpufortrt_map_create_r8_3,&
      gpufortrt_map_create_r8_4,&
      gpufortrt_map_create_r8_5,&
      gpufortrt_map_create_r8_6,&
      gpufortrt_map_create_r8_7,&
      gpufortrt_map_create_c4_0,&
      gpufortrt_map_create_c4_1,&
      gpufortrt_map_create_c4_2,&
      gpufortrt_map_create_c4_3,&
      gpufortrt_map_create_c4_4,&
      gpufortrt_map_create_c4_5,&
      gpufortrt_map_create_c4_6,&
      gpufortrt_map_create_c4_7,&
      gpufortrt_map_create_c8_0,&
      gpufortrt_map_create_c8_1,&
      gpufortrt_map_create_c8_2,&
      gpufortrt_map_create_c8_3,&
      gpufortrt_map_create_c8_4,&
      gpufortrt_map_create_c8_5,&
      gpufortrt_map_create_c8_6,&
      gpufortrt_map_create_c8_7
  end interface


 
  interface gpufortrt_map_copy
    module procedure &
      gpufortrt_map_copy_b, &
      gpufortrt_map_copy_l_0,&
      gpufortrt_map_copy_l_1,&
      gpufortrt_map_copy_l_2,&
      gpufortrt_map_copy_l_3,&
      gpufortrt_map_copy_l_4,&
      gpufortrt_map_copy_l_5,&
      gpufortrt_map_copy_l_6,&
      gpufortrt_map_copy_l_7,&
      gpufortrt_map_copy_c_0,&
      gpufortrt_map_copy_c_1,&
      gpufortrt_map_copy_c_2,&
      gpufortrt_map_copy_c_3,&
      gpufortrt_map_copy_c_4,&
      gpufortrt_map_copy_c_5,&
      gpufortrt_map_copy_c_6,&
      gpufortrt_map_copy_c_7,&
      gpufortrt_map_copy_i2_0,&
      gpufortrt_map_copy_i2_1,&
      gpufortrt_map_copy_i2_2,&
      gpufortrt_map_copy_i2_3,&
      gpufortrt_map_copy_i2_4,&
      gpufortrt_map_copy_i2_5,&
      gpufortrt_map_copy_i2_6,&
      gpufortrt_map_copy_i2_7,&
      gpufortrt_map_copy_i4_0,&
      gpufortrt_map_copy_i4_1,&
      gpufortrt_map_copy_i4_2,&
      gpufortrt_map_copy_i4_3,&
      gpufortrt_map_copy_i4_4,&
      gpufortrt_map_copy_i4_5,&
      gpufortrt_map_copy_i4_6,&
      gpufortrt_map_copy_i4_7,&
      gpufortrt_map_copy_i8_0,&
      gpufortrt_map_copy_i8_1,&
      gpufortrt_map_copy_i8_2,&
      gpufortrt_map_copy_i8_3,&
      gpufortrt_map_copy_i8_4,&
      gpufortrt_map_copy_i8_5,&
      gpufortrt_map_copy_i8_6,&
      gpufortrt_map_copy_i8_7,&
      gpufortrt_map_copy_r4_0,&
      gpufortrt_map_copy_r4_1,&
      gpufortrt_map_copy_r4_2,&
      gpufortrt_map_copy_r4_3,&
      gpufortrt_map_copy_r4_4,&
      gpufortrt_map_copy_r4_5,&
      gpufortrt_map_copy_r4_6,&
      gpufortrt_map_copy_r4_7,&
      gpufortrt_map_copy_r8_0,&
      gpufortrt_map_copy_r8_1,&
      gpufortrt_map_copy_r8_2,&
      gpufortrt_map_copy_r8_3,&
      gpufortrt_map_copy_r8_4,&
      gpufortrt_map_copy_r8_5,&
      gpufortrt_map_copy_r8_6,&
      gpufortrt_map_copy_r8_7,&
      gpufortrt_map_copy_c4_0,&
      gpufortrt_map_copy_c4_1,&
      gpufortrt_map_copy_c4_2,&
      gpufortrt_map_copy_c4_3,&
      gpufortrt_map_copy_c4_4,&
      gpufortrt_map_copy_c4_5,&
      gpufortrt_map_copy_c4_6,&
      gpufortrt_map_copy_c4_7,&
      gpufortrt_map_copy_c8_0,&
      gpufortrt_map_copy_c8_1,&
      gpufortrt_map_copy_c8_2,&
      gpufortrt_map_copy_c8_3,&
      gpufortrt_map_copy_c8_4,&
      gpufortrt_map_copy_c8_5,&
      gpufortrt_map_copy_c8_6,&
      gpufortrt_map_copy_c8_7
  end interface


 
  interface gpufortrt_map_copyin
    module procedure &
      gpufortrt_map_copyin_b, &
      gpufortrt_map_copyin_l_0,&
      gpufortrt_map_copyin_l_1,&
      gpufortrt_map_copyin_l_2,&
      gpufortrt_map_copyin_l_3,&
      gpufortrt_map_copyin_l_4,&
      gpufortrt_map_copyin_l_5,&
      gpufortrt_map_copyin_l_6,&
      gpufortrt_map_copyin_l_7,&
      gpufortrt_map_copyin_c_0,&
      gpufortrt_map_copyin_c_1,&
      gpufortrt_map_copyin_c_2,&
      gpufortrt_map_copyin_c_3,&
      gpufortrt_map_copyin_c_4,&
      gpufortrt_map_copyin_c_5,&
      gpufortrt_map_copyin_c_6,&
      gpufortrt_map_copyin_c_7,&
      gpufortrt_map_copyin_i2_0,&
      gpufortrt_map_copyin_i2_1,&
      gpufortrt_map_copyin_i2_2,&
      gpufortrt_map_copyin_i2_3,&
      gpufortrt_map_copyin_i2_4,&
      gpufortrt_map_copyin_i2_5,&
      gpufortrt_map_copyin_i2_6,&
      gpufortrt_map_copyin_i2_7,&
      gpufortrt_map_copyin_i4_0,&
      gpufortrt_map_copyin_i4_1,&
      gpufortrt_map_copyin_i4_2,&
      gpufortrt_map_copyin_i4_3,&
      gpufortrt_map_copyin_i4_4,&
      gpufortrt_map_copyin_i4_5,&
      gpufortrt_map_copyin_i4_6,&
      gpufortrt_map_copyin_i4_7,&
      gpufortrt_map_copyin_i8_0,&
      gpufortrt_map_copyin_i8_1,&
      gpufortrt_map_copyin_i8_2,&
      gpufortrt_map_copyin_i8_3,&
      gpufortrt_map_copyin_i8_4,&
      gpufortrt_map_copyin_i8_5,&
      gpufortrt_map_copyin_i8_6,&
      gpufortrt_map_copyin_i8_7,&
      gpufortrt_map_copyin_r4_0,&
      gpufortrt_map_copyin_r4_1,&
      gpufortrt_map_copyin_r4_2,&
      gpufortrt_map_copyin_r4_3,&
      gpufortrt_map_copyin_r4_4,&
      gpufortrt_map_copyin_r4_5,&
      gpufortrt_map_copyin_r4_6,&
      gpufortrt_map_copyin_r4_7,&
      gpufortrt_map_copyin_r8_0,&
      gpufortrt_map_copyin_r8_1,&
      gpufortrt_map_copyin_r8_2,&
      gpufortrt_map_copyin_r8_3,&
      gpufortrt_map_copyin_r8_4,&
      gpufortrt_map_copyin_r8_5,&
      gpufortrt_map_copyin_r8_6,&
      gpufortrt_map_copyin_r8_7,&
      gpufortrt_map_copyin_c4_0,&
      gpufortrt_map_copyin_c4_1,&
      gpufortrt_map_copyin_c4_2,&
      gpufortrt_map_copyin_c4_3,&
      gpufortrt_map_copyin_c4_4,&
      gpufortrt_map_copyin_c4_5,&
      gpufortrt_map_copyin_c4_6,&
      gpufortrt_map_copyin_c4_7,&
      gpufortrt_map_copyin_c8_0,&
      gpufortrt_map_copyin_c8_1,&
      gpufortrt_map_copyin_c8_2,&
      gpufortrt_map_copyin_c8_3,&
      gpufortrt_map_copyin_c8_4,&
      gpufortrt_map_copyin_c8_5,&
      gpufortrt_map_copyin_c8_6,&
      gpufortrt_map_copyin_c8_7
  end interface


 
  interface gpufortrt_map_copyout
    module procedure &
      gpufortrt_map_copyout_b, &
      gpufortrt_map_copyout_l_0,&
      gpufortrt_map_copyout_l_1,&
      gpufortrt_map_copyout_l_2,&
      gpufortrt_map_copyout_l_3,&
      gpufortrt_map_copyout_l_4,&
      gpufortrt_map_copyout_l_5,&
      gpufortrt_map_copyout_l_6,&
      gpufortrt_map_copyout_l_7,&
      gpufortrt_map_copyout_c_0,&
      gpufortrt_map_copyout_c_1,&
      gpufortrt_map_copyout_c_2,&
      gpufortrt_map_copyout_c_3,&
      gpufortrt_map_copyout_c_4,&
      gpufortrt_map_copyout_c_5,&
      gpufortrt_map_copyout_c_6,&
      gpufortrt_map_copyout_c_7,&
      gpufortrt_map_copyout_i2_0,&
      gpufortrt_map_copyout_i2_1,&
      gpufortrt_map_copyout_i2_2,&
      gpufortrt_map_copyout_i2_3,&
      gpufortrt_map_copyout_i2_4,&
      gpufortrt_map_copyout_i2_5,&
      gpufortrt_map_copyout_i2_6,&
      gpufortrt_map_copyout_i2_7,&
      gpufortrt_map_copyout_i4_0,&
      gpufortrt_map_copyout_i4_1,&
      gpufortrt_map_copyout_i4_2,&
      gpufortrt_map_copyout_i4_3,&
      gpufortrt_map_copyout_i4_4,&
      gpufortrt_map_copyout_i4_5,&
      gpufortrt_map_copyout_i4_6,&
      gpufortrt_map_copyout_i4_7,&
      gpufortrt_map_copyout_i8_0,&
      gpufortrt_map_copyout_i8_1,&
      gpufortrt_map_copyout_i8_2,&
      gpufortrt_map_copyout_i8_3,&
      gpufortrt_map_copyout_i8_4,&
      gpufortrt_map_copyout_i8_5,&
      gpufortrt_map_copyout_i8_6,&
      gpufortrt_map_copyout_i8_7,&
      gpufortrt_map_copyout_r4_0,&
      gpufortrt_map_copyout_r4_1,&
      gpufortrt_map_copyout_r4_2,&
      gpufortrt_map_copyout_r4_3,&
      gpufortrt_map_copyout_r4_4,&
      gpufortrt_map_copyout_r4_5,&
      gpufortrt_map_copyout_r4_6,&
      gpufortrt_map_copyout_r4_7,&
      gpufortrt_map_copyout_r8_0,&
      gpufortrt_map_copyout_r8_1,&
      gpufortrt_map_copyout_r8_2,&
      gpufortrt_map_copyout_r8_3,&
      gpufortrt_map_copyout_r8_4,&
      gpufortrt_map_copyout_r8_5,&
      gpufortrt_map_copyout_r8_6,&
      gpufortrt_map_copyout_r8_7,&
      gpufortrt_map_copyout_c4_0,&
      gpufortrt_map_copyout_c4_1,&
      gpufortrt_map_copyout_c4_2,&
      gpufortrt_map_copyout_c4_3,&
      gpufortrt_map_copyout_c4_4,&
      gpufortrt_map_copyout_c4_5,&
      gpufortrt_map_copyout_c4_6,&
      gpufortrt_map_copyout_c4_7,&
      gpufortrt_map_copyout_c8_0,&
      gpufortrt_map_copyout_c8_1,&
      gpufortrt_map_copyout_c8_2,&
      gpufortrt_map_copyout_c8_3,&
      gpufortrt_map_copyout_c8_4,&
      gpufortrt_map_copyout_c8_5,&
      gpufortrt_map_copyout_c8_6,&
      gpufortrt_map_copyout_c8_7
  end interface



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
  !> wait [( expression-list )]
  !>   The data movement will not begin execution until all actions on
  !>   the corresponding async queue(s) are complete.
  interface gpufortrt_update_host
    module procedure &
      gpufortrt_update_host_b, &
      gpufortrt_update_host_l_0,&
      gpufortrt_update_host_l_1,&
      gpufortrt_update_host_l_2,&
      gpufortrt_update_host_l_3,&
      gpufortrt_update_host_l_4,&
      gpufortrt_update_host_l_5,&
      gpufortrt_update_host_l_6,&
      gpufortrt_update_host_l_7,&
      gpufortrt_update_host_c_0,&
      gpufortrt_update_host_c_1,&
      gpufortrt_update_host_c_2,&
      gpufortrt_update_host_c_3,&
      gpufortrt_update_host_c_4,&
      gpufortrt_update_host_c_5,&
      gpufortrt_update_host_c_6,&
      gpufortrt_update_host_c_7,&
      gpufortrt_update_host_i2_0,&
      gpufortrt_update_host_i2_1,&
      gpufortrt_update_host_i2_2,&
      gpufortrt_update_host_i2_3,&
      gpufortrt_update_host_i2_4,&
      gpufortrt_update_host_i2_5,&
      gpufortrt_update_host_i2_6,&
      gpufortrt_update_host_i2_7,&
      gpufortrt_update_host_i4_0,&
      gpufortrt_update_host_i4_1,&
      gpufortrt_update_host_i4_2,&
      gpufortrt_update_host_i4_3,&
      gpufortrt_update_host_i4_4,&
      gpufortrt_update_host_i4_5,&
      gpufortrt_update_host_i4_6,&
      gpufortrt_update_host_i4_7,&
      gpufortrt_update_host_i8_0,&
      gpufortrt_update_host_i8_1,&
      gpufortrt_update_host_i8_2,&
      gpufortrt_update_host_i8_3,&
      gpufortrt_update_host_i8_4,&
      gpufortrt_update_host_i8_5,&
      gpufortrt_update_host_i8_6,&
      gpufortrt_update_host_i8_7,&
      gpufortrt_update_host_r4_0,&
      gpufortrt_update_host_r4_1,&
      gpufortrt_update_host_r4_2,&
      gpufortrt_update_host_r4_3,&
      gpufortrt_update_host_r4_4,&
      gpufortrt_update_host_r4_5,&
      gpufortrt_update_host_r4_6,&
      gpufortrt_update_host_r4_7,&
      gpufortrt_update_host_r8_0,&
      gpufortrt_update_host_r8_1,&
      gpufortrt_update_host_r8_2,&
      gpufortrt_update_host_r8_3,&
      gpufortrt_update_host_r8_4,&
      gpufortrt_update_host_r8_5,&
      gpufortrt_update_host_r8_6,&
      gpufortrt_update_host_r8_7,&
      gpufortrt_update_host_c4_0,&
      gpufortrt_update_host_c4_1,&
      gpufortrt_update_host_c4_2,&
      gpufortrt_update_host_c4_3,&
      gpufortrt_update_host_c4_4,&
      gpufortrt_update_host_c4_5,&
      gpufortrt_update_host_c4_6,&
      gpufortrt_update_host_c4_7,&
      gpufortrt_update_host_c8_0,&
      gpufortrt_update_host_c8_1,&
      gpufortrt_update_host_c8_2,&
      gpufortrt_update_host_c8_3,&
      gpufortrt_update_host_c8_4,&
      gpufortrt_update_host_c8_5,&
      gpufortrt_update_host_c8_6,&
      gpufortrt_update_host_c8_7
  end interface


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
  !> wait [( expression-list )]
  !>   The data movement will not begin execution until all actions on
  !>   the corresponding async queue(s) are complete.
  interface gpufortrt_update_device
    module procedure &
      gpufortrt_update_device_b, &
      gpufortrt_update_device_l_0,&
      gpufortrt_update_device_l_1,&
      gpufortrt_update_device_l_2,&
      gpufortrt_update_device_l_3,&
      gpufortrt_update_device_l_4,&
      gpufortrt_update_device_l_5,&
      gpufortrt_update_device_l_6,&
      gpufortrt_update_device_l_7,&
      gpufortrt_update_device_c_0,&
      gpufortrt_update_device_c_1,&
      gpufortrt_update_device_c_2,&
      gpufortrt_update_device_c_3,&
      gpufortrt_update_device_c_4,&
      gpufortrt_update_device_c_5,&
      gpufortrt_update_device_c_6,&
      gpufortrt_update_device_c_7,&
      gpufortrt_update_device_i2_0,&
      gpufortrt_update_device_i2_1,&
      gpufortrt_update_device_i2_2,&
      gpufortrt_update_device_i2_3,&
      gpufortrt_update_device_i2_4,&
      gpufortrt_update_device_i2_5,&
      gpufortrt_update_device_i2_6,&
      gpufortrt_update_device_i2_7,&
      gpufortrt_update_device_i4_0,&
      gpufortrt_update_device_i4_1,&
      gpufortrt_update_device_i4_2,&
      gpufortrt_update_device_i4_3,&
      gpufortrt_update_device_i4_4,&
      gpufortrt_update_device_i4_5,&
      gpufortrt_update_device_i4_6,&
      gpufortrt_update_device_i4_7,&
      gpufortrt_update_device_i8_0,&
      gpufortrt_update_device_i8_1,&
      gpufortrt_update_device_i8_2,&
      gpufortrt_update_device_i8_3,&
      gpufortrt_update_device_i8_4,&
      gpufortrt_update_device_i8_5,&
      gpufortrt_update_device_i8_6,&
      gpufortrt_update_device_i8_7,&
      gpufortrt_update_device_r4_0,&
      gpufortrt_update_device_r4_1,&
      gpufortrt_update_device_r4_2,&
      gpufortrt_update_device_r4_3,&
      gpufortrt_update_device_r4_4,&
      gpufortrt_update_device_r4_5,&
      gpufortrt_update_device_r4_6,&
      gpufortrt_update_device_r4_7,&
      gpufortrt_update_device_r8_0,&
      gpufortrt_update_device_r8_1,&
      gpufortrt_update_device_r8_2,&
      gpufortrt_update_device_r8_3,&
      gpufortrt_update_device_r8_4,&
      gpufortrt_update_device_r8_5,&
      gpufortrt_update_device_r8_6,&
      gpufortrt_update_device_r8_7,&
      gpufortrt_update_device_c4_0,&
      gpufortrt_update_device_c4_1,&
      gpufortrt_update_device_c4_2,&
      gpufortrt_update_device_c4_3,&
      gpufortrt_update_device_c4_4,&
      gpufortrt_update_device_c4_5,&
      gpufortrt_update_device_c4_6,&
      gpufortrt_update_device_c4_7,&
      gpufortrt_update_device_c8_0,&
      gpufortrt_update_device_c8_1,&
      gpufortrt_update_device_c8_2,&
      gpufortrt_update_device_c8_3,&
      gpufortrt_update_device_c8_4,&
      gpufortrt_update_device_c8_5,&
      gpufortrt_update_device_c8_6,&
      gpufortrt_update_device_c8_7
  end interface


  contains

    !
    ! autogenerated routines for different inputs
    !

    function gpufortrt_use_device_l_0(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      logical,target,intent(in) :: hostptr
      integer(c_int),dimension(0),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      logical,pointer :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),1_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr)
    end function 

    function gpufortrt_present_l_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      logical,target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),1_c_size_t)
    end function

    function gpufortrt_create_l_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      logical,target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),1_c_size_t)
    end function

    function gpufortrt_no_create_l_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      logical,target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_l_0(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      logical,target,intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_l_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      logical,target,intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),1_c_size_t,async)
    end function

    function gpufortrt_copyout_l_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      logical,target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),1_c_size_t,async)
    end function

    function gpufortrt_copy_l_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      logical,target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),1_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_l_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      logical,target,intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_l_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      logical,target,intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_l_1(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      logical,target,dimension(:),intent(in) :: hostptr
      integer(c_int),dimension(1),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      logical,pointer,dimension(:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*1_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_l_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      logical,target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_create_l_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      logical,target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_no_create_l_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      logical,target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_l_1(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      logical,target,dimension(:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_l_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      logical,target,dimension(:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copyout_l_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      logical,target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copy_l_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      logical,target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_l_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      logical,target,dimension(:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_l_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      logical,target,dimension(:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_l_2(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      logical,target,dimension(:,:),intent(in) :: hostptr
      integer(c_int),dimension(2),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      logical,pointer,dimension(:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*1_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_l_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      logical,target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_create_l_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      logical,target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_no_create_l_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      logical,target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_l_2(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      logical,target,dimension(:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_l_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      logical,target,dimension(:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copyout_l_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      logical,target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copy_l_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      logical,target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_l_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      logical,target,dimension(:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_l_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      logical,target,dimension(:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_l_3(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      logical,target,dimension(:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(3),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      logical,pointer,dimension(:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*1_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_l_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      logical,target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_create_l_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      logical,target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_no_create_l_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      logical,target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_l_3(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      logical,target,dimension(:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_l_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      logical,target,dimension(:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copyout_l_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      logical,target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copy_l_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      logical,target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_l_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      logical,target,dimension(:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_l_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      logical,target,dimension(:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_l_4(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      logical,target,dimension(:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(4),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      logical,pointer,dimension(:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*1_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_l_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      logical,target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_create_l_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      logical,target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_no_create_l_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      logical,target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_l_4(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      logical,target,dimension(:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_l_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      logical,target,dimension(:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copyout_l_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      logical,target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copy_l_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      logical,target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_l_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      logical,target,dimension(:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_l_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      logical,target,dimension(:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_l_5(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      logical,target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(5),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      logical,pointer,dimension(:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*1_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_l_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      logical,target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_create_l_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      logical,target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_no_create_l_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      logical,target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_l_5(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      logical,target,dimension(:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_l_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      logical,target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copyout_l_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      logical,target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copy_l_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      logical,target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_l_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      logical,target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_l_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      logical,target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_l_6(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(6),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      logical,pointer,dimension(:,:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*1_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):,&
          lbound(hostptr,6):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_l_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      logical,target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_create_l_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_no_create_l_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_l_6(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_l_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copyout_l_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copy_l_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_l_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_l_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_l_7(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(7),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      logical,pointer,dimension(:,:,:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*1_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):,&
          lbound(hostptr,6):,&
          lbound(hostptr,7):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_l_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      logical,target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_create_l_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_no_create_l_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_l_7(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_l_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copyout_l_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copy_l_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_l_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_l_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      logical,target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_c_0(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      character(c_char),target,intent(in) :: hostptr
      integer(c_int),dimension(0),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      character(c_char),pointer :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),1_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr)
    end function 

    function gpufortrt_present_c_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      character(c_char),target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),1_c_size_t)
    end function

    function gpufortrt_create_c_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      character(c_char),target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),1_c_size_t)
    end function

    function gpufortrt_no_create_c_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      character(c_char),target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_c_0(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      character(c_char),target,intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_c_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      character(c_char),target,intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),1_c_size_t,async)
    end function

    function gpufortrt_copyout_c_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      character(c_char),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),1_c_size_t,async)
    end function

    function gpufortrt_copy_c_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      character(c_char),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),1_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_c_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      character(c_char),target,intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_c_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      character(c_char),target,intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_c_1(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      character(c_char),target,dimension(:),intent(in) :: hostptr
      integer(c_int),dimension(1),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      character(c_char),pointer,dimension(:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*1_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_c_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      character(c_char),target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_create_c_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      character(c_char),target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_no_create_c_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      character(c_char),target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_c_1(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      character(c_char),target,dimension(:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_c_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      character(c_char),target,dimension(:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copyout_c_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      character(c_char),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copy_c_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      character(c_char),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_c_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      character(c_char),target,dimension(:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_c_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      character(c_char),target,dimension(:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_c_2(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      character(c_char),target,dimension(:,:),intent(in) :: hostptr
      integer(c_int),dimension(2),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      character(c_char),pointer,dimension(:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*1_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_c_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      character(c_char),target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_create_c_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      character(c_char),target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_no_create_c_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      character(c_char),target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_c_2(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      character(c_char),target,dimension(:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_c_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      character(c_char),target,dimension(:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copyout_c_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      character(c_char),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copy_c_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      character(c_char),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_c_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      character(c_char),target,dimension(:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_c_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      character(c_char),target,dimension(:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_c_3(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      character(c_char),target,dimension(:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(3),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      character(c_char),pointer,dimension(:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*1_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_c_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      character(c_char),target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_create_c_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      character(c_char),target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_no_create_c_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      character(c_char),target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_c_3(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      character(c_char),target,dimension(:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_c_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      character(c_char),target,dimension(:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copyout_c_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      character(c_char),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copy_c_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      character(c_char),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_c_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      character(c_char),target,dimension(:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_c_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      character(c_char),target,dimension(:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_c_4(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      character(c_char),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(4),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      character(c_char),pointer,dimension(:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*1_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_c_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      character(c_char),target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_create_c_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      character(c_char),target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_no_create_c_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      character(c_char),target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_c_4(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      character(c_char),target,dimension(:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_c_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      character(c_char),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copyout_c_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      character(c_char),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copy_c_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      character(c_char),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_c_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      character(c_char),target,dimension(:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_c_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      character(c_char),target,dimension(:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_c_5(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(5),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      character(c_char),pointer,dimension(:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*1_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_c_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      character(c_char),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_create_c_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_no_create_c_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_c_5(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_c_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copyout_c_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copy_c_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_c_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_c_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_c_6(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(6),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      character(c_char),pointer,dimension(:,:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*1_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):,&
          lbound(hostptr,6):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_c_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      character(c_char),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_create_c_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_no_create_c_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_c_6(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_c_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copyout_c_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copy_c_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_c_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_c_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_c_7(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(7),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      character(c_char),pointer,dimension(:,:,:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*1_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):,&
          lbound(hostptr,6):,&
          lbound(hostptr,7):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_c_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      character(c_char),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_create_c_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*1_c_size_t)
    end function

    function gpufortrt_no_create_c_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_c_7(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_c_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copyout_c_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    function gpufortrt_copy_c_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*1_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_c_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_c_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      character(c_char),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_i2_0(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      integer(c_short),target,intent(in) :: hostptr
      integer(c_int),dimension(0),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      integer(c_short),pointer :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),2_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr)
    end function 

    function gpufortrt_present_i2_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      integer(c_short),target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),2_c_size_t)
    end function

    function gpufortrt_create_i2_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      integer(c_short),target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),2_c_size_t)
    end function

    function gpufortrt_no_create_i2_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      integer(c_short),target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_i2_0(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      integer(c_short),target,intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_i2_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      integer(c_short),target,intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),2_c_size_t,async)
    end function

    function gpufortrt_copyout_i2_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      integer(c_short),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),2_c_size_t,async)
    end function

    function gpufortrt_copy_i2_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      integer(c_short),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),2_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_i2_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_short),target,intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_i2_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_short),target,intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_i2_1(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      integer(c_short),target,dimension(:),intent(in) :: hostptr
      integer(c_int),dimension(1),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      integer(c_short),pointer,dimension(:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*2_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_i2_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      integer(c_short),target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*2_c_size_t)
    end function

    function gpufortrt_create_i2_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      integer(c_short),target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*2_c_size_t)
    end function

    function gpufortrt_no_create_i2_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      integer(c_short),target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_i2_1(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      integer(c_short),target,dimension(:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_i2_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      integer(c_short),target,dimension(:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*2_c_size_t,async)
    end function

    function gpufortrt_copyout_i2_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      integer(c_short),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*2_c_size_t,async)
    end function

    function gpufortrt_copy_i2_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      integer(c_short),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*2_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_i2_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_short),target,dimension(:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_i2_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_short),target,dimension(:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_i2_2(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      integer(c_short),target,dimension(:,:),intent(in) :: hostptr
      integer(c_int),dimension(2),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      integer(c_short),pointer,dimension(:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*2_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_i2_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      integer(c_short),target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*2_c_size_t)
    end function

    function gpufortrt_create_i2_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      integer(c_short),target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*2_c_size_t)
    end function

    function gpufortrt_no_create_i2_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      integer(c_short),target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_i2_2(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      integer(c_short),target,dimension(:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_i2_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      integer(c_short),target,dimension(:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*2_c_size_t,async)
    end function

    function gpufortrt_copyout_i2_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      integer(c_short),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*2_c_size_t,async)
    end function

    function gpufortrt_copy_i2_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      integer(c_short),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*2_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_i2_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_short),target,dimension(:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_i2_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_short),target,dimension(:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_i2_3(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      integer(c_short),target,dimension(:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(3),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      integer(c_short),pointer,dimension(:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*2_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_i2_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      integer(c_short),target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*2_c_size_t)
    end function

    function gpufortrt_create_i2_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      integer(c_short),target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*2_c_size_t)
    end function

    function gpufortrt_no_create_i2_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      integer(c_short),target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_i2_3(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      integer(c_short),target,dimension(:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_i2_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      integer(c_short),target,dimension(:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*2_c_size_t,async)
    end function

    function gpufortrt_copyout_i2_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      integer(c_short),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*2_c_size_t,async)
    end function

    function gpufortrt_copy_i2_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      integer(c_short),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*2_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_i2_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_short),target,dimension(:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_i2_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_short),target,dimension(:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_i2_4(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(4),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      integer(c_short),pointer,dimension(:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*2_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_i2_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      integer(c_short),target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*2_c_size_t)
    end function

    function gpufortrt_create_i2_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*2_c_size_t)
    end function

    function gpufortrt_no_create_i2_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_i2_4(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_i2_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*2_c_size_t,async)
    end function

    function gpufortrt_copyout_i2_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*2_c_size_t,async)
    end function

    function gpufortrt_copy_i2_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*2_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_i2_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_i2_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_i2_5(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(5),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      integer(c_short),pointer,dimension(:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*2_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_i2_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*2_c_size_t)
    end function

    function gpufortrt_create_i2_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*2_c_size_t)
    end function

    function gpufortrt_no_create_i2_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_i2_5(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_i2_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*2_c_size_t,async)
    end function

    function gpufortrt_copyout_i2_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*2_c_size_t,async)
    end function

    function gpufortrt_copy_i2_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*2_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_i2_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_i2_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_i2_6(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(6),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      integer(c_short),pointer,dimension(:,:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*2_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):,&
          lbound(hostptr,6):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_i2_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*2_c_size_t)
    end function

    function gpufortrt_create_i2_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*2_c_size_t)
    end function

    function gpufortrt_no_create_i2_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_i2_6(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_i2_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*2_c_size_t,async)
    end function

    function gpufortrt_copyout_i2_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*2_c_size_t,async)
    end function

    function gpufortrt_copy_i2_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*2_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_i2_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_i2_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_i2_7(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(7),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      integer(c_short),pointer,dimension(:,:,:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*2_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):,&
          lbound(hostptr,6):,&
          lbound(hostptr,7):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_i2_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*2_c_size_t)
    end function

    function gpufortrt_create_i2_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*2_c_size_t)
    end function

    function gpufortrt_no_create_i2_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_i2_7(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_i2_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*2_c_size_t,async)
    end function

    function gpufortrt_copyout_i2_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*2_c_size_t,async)
    end function

    function gpufortrt_copy_i2_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*2_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_i2_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_i2_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_short),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_i4_0(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      integer(c_int),target,intent(in) :: hostptr
      integer(c_int),dimension(0),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      integer(c_int),pointer :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),4_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr)
    end function 

    function gpufortrt_present_i4_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      integer(c_int),target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),4_c_size_t)
    end function

    function gpufortrt_create_i4_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      integer(c_int),target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),4_c_size_t)
    end function

    function gpufortrt_no_create_i4_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      integer(c_int),target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_i4_0(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      integer(c_int),target,intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_i4_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      integer(c_int),target,intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),4_c_size_t,async)
    end function

    function gpufortrt_copyout_i4_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      integer(c_int),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),4_c_size_t,async)
    end function

    function gpufortrt_copy_i4_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      integer(c_int),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),4_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_i4_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_int),target,intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_i4_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_int),target,intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_i4_1(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      integer(c_int),target,dimension(:),intent(in) :: hostptr
      integer(c_int),dimension(1),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      integer(c_int),pointer,dimension(:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*4_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_i4_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      integer(c_int),target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_create_i4_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      integer(c_int),target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_no_create_i4_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      integer(c_int),target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_i4_1(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      integer(c_int),target,dimension(:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_i4_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      integer(c_int),target,dimension(:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copyout_i4_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      integer(c_int),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copy_i4_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      integer(c_int),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_i4_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_int),target,dimension(:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_i4_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_int),target,dimension(:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_i4_2(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      integer(c_int),target,dimension(:,:),intent(in) :: hostptr
      integer(c_int),dimension(2),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      integer(c_int),pointer,dimension(:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*4_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_i4_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      integer(c_int),target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_create_i4_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      integer(c_int),target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_no_create_i4_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      integer(c_int),target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_i4_2(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      integer(c_int),target,dimension(:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_i4_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      integer(c_int),target,dimension(:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copyout_i4_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      integer(c_int),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copy_i4_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      integer(c_int),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_i4_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_int),target,dimension(:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_i4_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_int),target,dimension(:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_i4_3(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      integer(c_int),target,dimension(:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(3),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      integer(c_int),pointer,dimension(:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*4_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_i4_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      integer(c_int),target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_create_i4_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      integer(c_int),target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_no_create_i4_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      integer(c_int),target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_i4_3(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      integer(c_int),target,dimension(:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_i4_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      integer(c_int),target,dimension(:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copyout_i4_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      integer(c_int),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copy_i4_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      integer(c_int),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_i4_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_int),target,dimension(:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_i4_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_int),target,dimension(:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_i4_4(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(4),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      integer(c_int),pointer,dimension(:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*4_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_i4_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      integer(c_int),target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_create_i4_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_no_create_i4_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_i4_4(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_i4_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copyout_i4_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copy_i4_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_i4_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_i4_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_i4_5(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(5),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      integer(c_int),pointer,dimension(:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*4_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_i4_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_create_i4_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_no_create_i4_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_i4_5(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_i4_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copyout_i4_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copy_i4_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_i4_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_i4_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_i4_6(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(6),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      integer(c_int),pointer,dimension(:,:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*4_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):,&
          lbound(hostptr,6):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_i4_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_create_i4_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_no_create_i4_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_i4_6(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_i4_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copyout_i4_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copy_i4_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_i4_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_i4_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_i4_7(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(7),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      integer(c_int),pointer,dimension(:,:,:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*4_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):,&
          lbound(hostptr,6):,&
          lbound(hostptr,7):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_i4_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_create_i4_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_no_create_i4_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_i4_7(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_i4_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copyout_i4_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copy_i4_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_i4_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_i4_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_i8_0(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      integer(c_long),target,intent(in) :: hostptr
      integer(c_int),dimension(0),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      integer(c_long),pointer :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),8_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr)
    end function 

    function gpufortrt_present_i8_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      integer(c_long),target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),8_c_size_t)
    end function

    function gpufortrt_create_i8_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      integer(c_long),target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),8_c_size_t)
    end function

    function gpufortrt_no_create_i8_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      integer(c_long),target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_i8_0(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      integer(c_long),target,intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_i8_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      integer(c_long),target,intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),8_c_size_t,async)
    end function

    function gpufortrt_copyout_i8_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      integer(c_long),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),8_c_size_t,async)
    end function

    function gpufortrt_copy_i8_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      integer(c_long),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),8_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_i8_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_long),target,intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_i8_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_long),target,intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_i8_1(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      integer(c_long),target,dimension(:),intent(in) :: hostptr
      integer(c_int),dimension(1),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      integer(c_long),pointer,dimension(:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*8_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_i8_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      integer(c_long),target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_create_i8_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      integer(c_long),target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_no_create_i8_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      integer(c_long),target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_i8_1(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      integer(c_long),target,dimension(:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_i8_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      integer(c_long),target,dimension(:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copyout_i8_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      integer(c_long),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copy_i8_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      integer(c_long),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_i8_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_long),target,dimension(:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_i8_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_long),target,dimension(:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_i8_2(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      integer(c_long),target,dimension(:,:),intent(in) :: hostptr
      integer(c_int),dimension(2),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      integer(c_long),pointer,dimension(:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*8_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_i8_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      integer(c_long),target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_create_i8_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      integer(c_long),target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_no_create_i8_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      integer(c_long),target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_i8_2(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      integer(c_long),target,dimension(:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_i8_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      integer(c_long),target,dimension(:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copyout_i8_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      integer(c_long),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copy_i8_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      integer(c_long),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_i8_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_long),target,dimension(:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_i8_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_long),target,dimension(:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_i8_3(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      integer(c_long),target,dimension(:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(3),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      integer(c_long),pointer,dimension(:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*8_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_i8_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      integer(c_long),target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_create_i8_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      integer(c_long),target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_no_create_i8_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      integer(c_long),target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_i8_3(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      integer(c_long),target,dimension(:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_i8_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      integer(c_long),target,dimension(:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copyout_i8_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      integer(c_long),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copy_i8_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      integer(c_long),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_i8_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_long),target,dimension(:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_i8_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_long),target,dimension(:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_i8_4(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(4),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      integer(c_long),pointer,dimension(:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*8_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_i8_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      integer(c_long),target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_create_i8_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_no_create_i8_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_i8_4(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_i8_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copyout_i8_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copy_i8_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_i8_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_i8_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_i8_5(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(5),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      integer(c_long),pointer,dimension(:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*8_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_i8_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_create_i8_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_no_create_i8_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_i8_5(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_i8_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copyout_i8_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copy_i8_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_i8_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_i8_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_i8_6(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(6),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      integer(c_long),pointer,dimension(:,:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*8_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):,&
          lbound(hostptr,6):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_i8_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_create_i8_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_no_create_i8_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_i8_6(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_i8_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copyout_i8_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copy_i8_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_i8_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_i8_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_i8_7(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(7),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      integer(c_long),pointer,dimension(:,:,:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*8_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):,&
          lbound(hostptr,6):,&
          lbound(hostptr,7):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_i8_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_create_i8_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_no_create_i8_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_i8_7(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_i8_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copyout_i8_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copy_i8_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_i8_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_i8_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_r4_0(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      real(c_float),target,intent(in) :: hostptr
      integer(c_int),dimension(0),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      real(c_float),pointer :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),4_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr)
    end function 

    function gpufortrt_present_r4_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      real(c_float),target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),4_c_size_t)
    end function

    function gpufortrt_create_r4_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      real(c_float),target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),4_c_size_t)
    end function

    function gpufortrt_no_create_r4_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      real(c_float),target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_r4_0(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      real(c_float),target,intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_r4_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      real(c_float),target,intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),4_c_size_t,async)
    end function

    function gpufortrt_copyout_r4_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      real(c_float),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),4_c_size_t,async)
    end function

    function gpufortrt_copy_r4_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      real(c_float),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),4_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_r4_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_float),target,intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_r4_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_float),target,intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_r4_1(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      real(c_float),target,dimension(:),intent(in) :: hostptr
      integer(c_int),dimension(1),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      real(c_float),pointer,dimension(:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*4_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_r4_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      real(c_float),target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_create_r4_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      real(c_float),target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_no_create_r4_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      real(c_float),target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_r4_1(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      real(c_float),target,dimension(:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_r4_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      real(c_float),target,dimension(:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copyout_r4_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      real(c_float),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copy_r4_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      real(c_float),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_r4_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_float),target,dimension(:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_r4_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_float),target,dimension(:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_r4_2(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      real(c_float),target,dimension(:,:),intent(in) :: hostptr
      integer(c_int),dimension(2),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      real(c_float),pointer,dimension(:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*4_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_r4_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      real(c_float),target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_create_r4_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      real(c_float),target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_no_create_r4_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      real(c_float),target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_r4_2(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      real(c_float),target,dimension(:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_r4_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      real(c_float),target,dimension(:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copyout_r4_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      real(c_float),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copy_r4_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      real(c_float),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_r4_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_float),target,dimension(:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_r4_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_float),target,dimension(:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_r4_3(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      real(c_float),target,dimension(:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(3),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      real(c_float),pointer,dimension(:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*4_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_r4_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      real(c_float),target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_create_r4_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      real(c_float),target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_no_create_r4_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      real(c_float),target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_r4_3(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      real(c_float),target,dimension(:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_r4_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      real(c_float),target,dimension(:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copyout_r4_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      real(c_float),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copy_r4_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      real(c_float),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_r4_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_float),target,dimension(:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_r4_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_float),target,dimension(:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_r4_4(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      real(c_float),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(4),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      real(c_float),pointer,dimension(:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*4_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_r4_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      real(c_float),target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_create_r4_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      real(c_float),target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_no_create_r4_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      real(c_float),target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_r4_4(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      real(c_float),target,dimension(:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_r4_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      real(c_float),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copyout_r4_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      real(c_float),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copy_r4_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      real(c_float),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_r4_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_float),target,dimension(:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_r4_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_float),target,dimension(:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_r4_5(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(5),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      real(c_float),pointer,dimension(:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*4_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_r4_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      real(c_float),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_create_r4_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_no_create_r4_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_r4_5(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_r4_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copyout_r4_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copy_r4_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_r4_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_r4_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_r4_6(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(6),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      real(c_float),pointer,dimension(:,:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*4_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):,&
          lbound(hostptr,6):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_r4_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_create_r4_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_no_create_r4_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_r4_6(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_r4_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copyout_r4_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copy_r4_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_r4_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_r4_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_r4_7(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(7),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      real(c_float),pointer,dimension(:,:,:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*4_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):,&
          lbound(hostptr,6):,&
          lbound(hostptr,7):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_r4_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_create_r4_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*4_c_size_t)
    end function

    function gpufortrt_no_create_r4_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_r4_7(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_r4_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copyout_r4_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    function gpufortrt_copy_r4_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*4_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_r4_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_r4_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_float),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_r8_0(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      real(c_double),target,intent(in) :: hostptr
      integer(c_int),dimension(0),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      real(c_double),pointer :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),8_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr)
    end function 

    function gpufortrt_present_r8_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      real(c_double),target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),8_c_size_t)
    end function

    function gpufortrt_create_r8_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      real(c_double),target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),8_c_size_t)
    end function

    function gpufortrt_no_create_r8_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      real(c_double),target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_r8_0(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      real(c_double),target,intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_r8_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      real(c_double),target,intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),8_c_size_t,async)
    end function

    function gpufortrt_copyout_r8_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      real(c_double),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),8_c_size_t,async)
    end function

    function gpufortrt_copy_r8_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      real(c_double),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),8_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_r8_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_double),target,intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_r8_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_double),target,intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_r8_1(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      real(c_double),target,dimension(:),intent(in) :: hostptr
      integer(c_int),dimension(1),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      real(c_double),pointer,dimension(:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*8_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_r8_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      real(c_double),target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_create_r8_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      real(c_double),target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_no_create_r8_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      real(c_double),target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_r8_1(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      real(c_double),target,dimension(:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_r8_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      real(c_double),target,dimension(:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copyout_r8_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      real(c_double),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copy_r8_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      real(c_double),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_r8_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_double),target,dimension(:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_r8_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_double),target,dimension(:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_r8_2(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      real(c_double),target,dimension(:,:),intent(in) :: hostptr
      integer(c_int),dimension(2),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      real(c_double),pointer,dimension(:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*8_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_r8_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      real(c_double),target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_create_r8_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      real(c_double),target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_no_create_r8_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      real(c_double),target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_r8_2(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      real(c_double),target,dimension(:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_r8_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      real(c_double),target,dimension(:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copyout_r8_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      real(c_double),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copy_r8_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      real(c_double),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_r8_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_double),target,dimension(:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_r8_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_double),target,dimension(:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_r8_3(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      real(c_double),target,dimension(:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(3),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      real(c_double),pointer,dimension(:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*8_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_r8_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      real(c_double),target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_create_r8_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      real(c_double),target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_no_create_r8_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      real(c_double),target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_r8_3(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      real(c_double),target,dimension(:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_r8_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      real(c_double),target,dimension(:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copyout_r8_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      real(c_double),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copy_r8_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      real(c_double),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_r8_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_double),target,dimension(:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_r8_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_double),target,dimension(:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_r8_4(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      real(c_double),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(4),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      real(c_double),pointer,dimension(:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*8_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_r8_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      real(c_double),target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_create_r8_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      real(c_double),target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_no_create_r8_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      real(c_double),target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_r8_4(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      real(c_double),target,dimension(:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_r8_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      real(c_double),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copyout_r8_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      real(c_double),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copy_r8_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      real(c_double),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_r8_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_double),target,dimension(:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_r8_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_double),target,dimension(:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_r8_5(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(5),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      real(c_double),pointer,dimension(:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*8_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_r8_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      real(c_double),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_create_r8_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_no_create_r8_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_r8_5(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_r8_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copyout_r8_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copy_r8_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_r8_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_r8_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_r8_6(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(6),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      real(c_double),pointer,dimension(:,:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*8_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):,&
          lbound(hostptr,6):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_r8_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_create_r8_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_no_create_r8_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_r8_6(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_r8_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copyout_r8_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copy_r8_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_r8_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_r8_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_r8_7(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(7),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      real(c_double),pointer,dimension(:,:,:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*8_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):,&
          lbound(hostptr,6):,&
          lbound(hostptr,7):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_r8_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_create_r8_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*8_c_size_t)
    end function

    function gpufortrt_no_create_r8_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_r8_7(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_r8_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copyout_r8_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    function gpufortrt_copy_r8_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*8_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_r8_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_r8_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      real(c_double),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_c4_0(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      complex(c_float_complex),target,intent(in) :: hostptr
      integer(c_int),dimension(0),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      complex(c_float_complex),pointer :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),2*4_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr)
    end function 

    function gpufortrt_present_c4_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      complex(c_float_complex),target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),2*4_c_size_t)
    end function

    function gpufortrt_create_c4_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      complex(c_float_complex),target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),2*4_c_size_t)
    end function

    function gpufortrt_no_create_c4_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      complex(c_float_complex),target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_c4_0(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      complex(c_float_complex),target,intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_c4_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      complex(c_float_complex),target,intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),2*4_c_size_t,async)
    end function

    function gpufortrt_copyout_c4_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      complex(c_float_complex),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),2*4_c_size_t,async)
    end function

    function gpufortrt_copy_c4_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      complex(c_float_complex),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),2*4_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_c4_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_float_complex),target,intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_c4_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_float_complex),target,intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_c4_1(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      complex(c_float_complex),target,dimension(:),intent(in) :: hostptr
      integer(c_int),dimension(1),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      complex(c_float_complex),pointer,dimension(:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_c4_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      complex(c_float_complex),target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t)
    end function

    function gpufortrt_create_c4_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      complex(c_float_complex),target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t)
    end function

    function gpufortrt_no_create_c4_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      complex(c_float_complex),target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_c4_1(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      complex(c_float_complex),target,dimension(:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_c4_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      complex(c_float_complex),target,dimension(:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,async)
    end function

    function gpufortrt_copyout_c4_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      complex(c_float_complex),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,async)
    end function

    function gpufortrt_copy_c4_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      complex(c_float_complex),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_c4_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_float_complex),target,dimension(:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_c4_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_float_complex),target,dimension(:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_c4_2(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      complex(c_float_complex),target,dimension(:,:),intent(in) :: hostptr
      integer(c_int),dimension(2),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      complex(c_float_complex),pointer,dimension(:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_c4_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      complex(c_float_complex),target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t)
    end function

    function gpufortrt_create_c4_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      complex(c_float_complex),target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t)
    end function

    function gpufortrt_no_create_c4_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      complex(c_float_complex),target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_c4_2(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      complex(c_float_complex),target,dimension(:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_c4_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      complex(c_float_complex),target,dimension(:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,async)
    end function

    function gpufortrt_copyout_c4_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      complex(c_float_complex),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,async)
    end function

    function gpufortrt_copy_c4_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      complex(c_float_complex),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_c4_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_float_complex),target,dimension(:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_c4_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_float_complex),target,dimension(:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_c4_3(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(3),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      complex(c_float_complex),pointer,dimension(:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_c4_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      complex(c_float_complex),target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t)
    end function

    function gpufortrt_create_c4_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t)
    end function

    function gpufortrt_no_create_c4_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_c4_3(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_c4_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,async)
    end function

    function gpufortrt_copyout_c4_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,async)
    end function

    function gpufortrt_copy_c4_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_c4_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_c4_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_c4_4(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(4),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      complex(c_float_complex),pointer,dimension(:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_c4_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t)
    end function

    function gpufortrt_create_c4_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t)
    end function

    function gpufortrt_no_create_c4_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_c4_4(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_c4_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,async)
    end function

    function gpufortrt_copyout_c4_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,async)
    end function

    function gpufortrt_copy_c4_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_c4_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_c4_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_c4_5(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(5),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      complex(c_float_complex),pointer,dimension(:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_c4_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t)
    end function

    function gpufortrt_create_c4_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t)
    end function

    function gpufortrt_no_create_c4_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_c4_5(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_c4_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,async)
    end function

    function gpufortrt_copyout_c4_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,async)
    end function

    function gpufortrt_copy_c4_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_c4_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_c4_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_c4_6(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(6),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):,&
          lbound(hostptr,6):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_c4_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t)
    end function

    function gpufortrt_create_c4_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t)
    end function

    function gpufortrt_no_create_c4_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_c4_6(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_c4_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,async)
    end function

    function gpufortrt_copyout_c4_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,async)
    end function

    function gpufortrt_copy_c4_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_c4_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_c4_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_c4_7(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(7),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      complex(c_float_complex),pointer,dimension(:,:,:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):,&
          lbound(hostptr,6):,&
          lbound(hostptr,7):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_c4_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t)
    end function

    function gpufortrt_create_c4_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t)
    end function

    function gpufortrt_no_create_c4_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_c4_7(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_c4_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,async)
    end function

    function gpufortrt_copyout_c4_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,async)
    end function

    function gpufortrt_copy_c4_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*2*4_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_c4_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_c4_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_c8_0(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      complex(c_double_complex),target,intent(in) :: hostptr
      integer(c_int),dimension(0),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      complex(c_double_complex),pointer :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),2*8_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr)
    end function 

    function gpufortrt_present_c8_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      complex(c_double_complex),target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),2*8_c_size_t)
    end function

    function gpufortrt_create_c8_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      complex(c_double_complex),target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),2*8_c_size_t)
    end function

    function gpufortrt_no_create_c8_0(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      complex(c_double_complex),target,intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_c8_0(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      complex(c_double_complex),target,intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_c8_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      complex(c_double_complex),target,intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),2*8_c_size_t,async)
    end function

    function gpufortrt_copyout_c8_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      complex(c_double_complex),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),2*8_c_size_t,async)
    end function

    function gpufortrt_copy_c8_0(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      complex(c_double_complex),target,intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),2*8_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_c8_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_double_complex),target,intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_c8_0(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_double_complex),target,intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_c8_1(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      complex(c_double_complex),target,dimension(:),intent(in) :: hostptr
      integer(c_int),dimension(1),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      complex(c_double_complex),pointer,dimension(:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_c8_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      complex(c_double_complex),target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t)
    end function

    function gpufortrt_create_c8_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      complex(c_double_complex),target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t)
    end function

    function gpufortrt_no_create_c8_1(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      complex(c_double_complex),target,dimension(:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_c8_1(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      complex(c_double_complex),target,dimension(:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_c8_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      complex(c_double_complex),target,dimension(:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,async)
    end function

    function gpufortrt_copyout_c8_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      complex(c_double_complex),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,async)
    end function

    function gpufortrt_copy_c8_1(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      complex(c_double_complex),target,dimension(:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_c8_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_double_complex),target,dimension(:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_c8_1(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_double_complex),target,dimension(:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_c8_2(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      complex(c_double_complex),target,dimension(:,:),intent(in) :: hostptr
      integer(c_int),dimension(2),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      complex(c_double_complex),pointer,dimension(:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_c8_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      complex(c_double_complex),target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t)
    end function

    function gpufortrt_create_c8_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      complex(c_double_complex),target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t)
    end function

    function gpufortrt_no_create_c8_2(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      complex(c_double_complex),target,dimension(:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_c8_2(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      complex(c_double_complex),target,dimension(:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_c8_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      complex(c_double_complex),target,dimension(:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,async)
    end function

    function gpufortrt_copyout_c8_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      complex(c_double_complex),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,async)
    end function

    function gpufortrt_copy_c8_2(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      complex(c_double_complex),target,dimension(:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_c8_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_double_complex),target,dimension(:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_c8_2(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_double_complex),target,dimension(:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_c8_3(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(3),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      complex(c_double_complex),pointer,dimension(:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_c8_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      complex(c_double_complex),target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t)
    end function

    function gpufortrt_create_c8_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t)
    end function

    function gpufortrt_no_create_c8_3(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_c8_3(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_c8_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,async)
    end function

    function gpufortrt_copyout_c8_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,async)
    end function

    function gpufortrt_copy_c8_3(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_c8_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_c8_3(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_c8_4(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(4),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      complex(c_double_complex),pointer,dimension(:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_c8_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t)
    end function

    function gpufortrt_create_c8_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t)
    end function

    function gpufortrt_no_create_c8_4(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_c8_4(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_c8_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,async)
    end function

    function gpufortrt_copyout_c8_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,async)
    end function

    function gpufortrt_copy_c8_4(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_c8_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_c8_4(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_c8_5(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(5),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      complex(c_double_complex),pointer,dimension(:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_c8_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t)
    end function

    function gpufortrt_create_c8_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t)
    end function

    function gpufortrt_no_create_c8_5(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_c8_5(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_c8_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,async)
    end function

    function gpufortrt_copyout_c8_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,async)
    end function

    function gpufortrt_copy_c8_5(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_c8_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_c8_5(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_c8_6(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(6),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):,&
          lbound(hostptr,6):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_c8_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t)
    end function

    function gpufortrt_create_c8_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t)
    end function

    function gpufortrt_no_create_c8_6(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_c8_6(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_c8_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,async)
    end function

    function gpufortrt_copyout_c8_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,async)
    end function

    function gpufortrt_copy_c8_6(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_c8_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_c8_6(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    function gpufortrt_use_device_c8_7(hostptr,lbounds,condition,if_present) result(resultptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_use_device_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer(c_int),dimension(7),intent(in),optional :: lbounds
      logical,intent(in),optional  :: condition,if_present
      !
      complex(c_double_complex),pointer,dimension(:,:,:,:,:,:,:) :: resultptr
      ! 
      type(c_ptr) :: cptr
      !
      cptr = gpufortrt_use_device_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,condition,if_present)
      !
      call c_f_pointer(cptr,resultptr,shape(hostptr))
      if ( present(lbounds) ) then
        resultptr(&
          lbound(hostptr,1):,&
          lbound(hostptr,2):,&
          lbound(hostptr,3):,&
          lbound(hostptr,4):,&
          lbound(hostptr,5):,&
          lbound(hostptr,6):,&
          lbound(hostptr,7):)&
            => resultptr
      endif
    end function 

    function gpufortrt_present_c8_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_present_b, gpufortrt_map_kind_undefined
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_present_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t)
    end function

    function gpufortrt_create_c8_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_create_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_create_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t)
    end function

    function gpufortrt_no_create_c8_7(hostptr) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_no_create_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_no_create_b(c_loc(hostptr))
    end function

    subroutine gpufortrt_delete_c8_7(hostptr,finalize)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_delete_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      logical,intent(in),optional              :: finalize
      !
      call gpufortrt_delete_b(c_loc(hostptr),finalize)
    end subroutine

    function gpufortrt_copyin_c8_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyin_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyin_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,async)
    end function

    function gpufortrt_copyout_c8_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copyout_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copyout_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,async)
    end function

    function gpufortrt_copy_c8_7(hostptr,async) result(deviceptr)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_copy_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      integer,intent(in),optional :: async
      !
      type(c_ptr) :: deviceptr
      !
      deviceptr = gpufortrt_copy_b(c_loc(hostptr),size(hostptr)*2*8_c_size_t,async)
    end function

    subroutine gpufortrt_update_host_c8_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_host_b(c_loc(hostptr),condition,if_present,async)
    end subroutine
    subroutine gpufortrt_update_device_c8_7(hostptr,condition,if_present,async)
      use iso_c_binding
      use gpufortrt_core, only: gpufortrt_update_host_b
      implicit none
      complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(inout) :: hostptr
      logical,intent(in),optional :: condition, if_present
      integer,intent(in),optional :: async
      !
      call gpufortrt_update_device_b(c_loc(hostptr),condition,if_present,async)
    end subroutine

                                                                
  function gpufortrt_map_dec_struct_refs_l_scal(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    call retval%init(c_loc(hostptr),0_c_size_T,gpufortrt_map_kind_dec_struct_refs,.false.)
  end function

  function gpufortrt_map_dec_struct_refs_l_arr(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,dimension(*),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    call retval%init(c_loc(hostptr),0_c_size_t,gpufortrt_map_kind_dec_struct_refs,.false.)
  end function
                                                                
  function gpufortrt_map_dec_struct_refs_c_scal(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    call retval%init(c_loc(hostptr),0_c_size_T,gpufortrt_map_kind_dec_struct_refs,.false.)
  end function

  function gpufortrt_map_dec_struct_refs_c_arr(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),dimension(*),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    call retval%init(c_loc(hostptr),0_c_size_t,gpufortrt_map_kind_dec_struct_refs,.false.)
  end function
                                                                
  function gpufortrt_map_dec_struct_refs_i2_scal(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    call retval%init(c_loc(hostptr),0_c_size_T,gpufortrt_map_kind_dec_struct_refs,.false.)
  end function

  function gpufortrt_map_dec_struct_refs_i2_arr(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),dimension(*),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    call retval%init(c_loc(hostptr),0_c_size_t,gpufortrt_map_kind_dec_struct_refs,.false.)
  end function
                                                                
  function gpufortrt_map_dec_struct_refs_i4_scal(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    call retval%init(c_loc(hostptr),0_c_size_T,gpufortrt_map_kind_dec_struct_refs,.false.)
  end function

  function gpufortrt_map_dec_struct_refs_i4_arr(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),dimension(*),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    call retval%init(c_loc(hostptr),0_c_size_t,gpufortrt_map_kind_dec_struct_refs,.false.)
  end function
                                                                
  function gpufortrt_map_dec_struct_refs_i8_scal(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    call retval%init(c_loc(hostptr),0_c_size_T,gpufortrt_map_kind_dec_struct_refs,.false.)
  end function

  function gpufortrt_map_dec_struct_refs_i8_arr(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),dimension(*),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    call retval%init(c_loc(hostptr),0_c_size_t,gpufortrt_map_kind_dec_struct_refs,.false.)
  end function
                                                                
  function gpufortrt_map_dec_struct_refs_r4_scal(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    call retval%init(c_loc(hostptr),0_c_size_T,gpufortrt_map_kind_dec_struct_refs,.false.)
  end function

  function gpufortrt_map_dec_struct_refs_r4_arr(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),dimension(*),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    call retval%init(c_loc(hostptr),0_c_size_t,gpufortrt_map_kind_dec_struct_refs,.false.)
  end function
                                                                
  function gpufortrt_map_dec_struct_refs_r8_scal(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    call retval%init(c_loc(hostptr),0_c_size_T,gpufortrt_map_kind_dec_struct_refs,.false.)
  end function

  function gpufortrt_map_dec_struct_refs_r8_arr(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),dimension(*),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    call retval%init(c_loc(hostptr),0_c_size_t,gpufortrt_map_kind_dec_struct_refs,.false.)
  end function
                                                                
  function gpufortrt_map_dec_struct_refs_c4_scal(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    call retval%init(c_loc(hostptr),0_c_size_T,gpufortrt_map_kind_dec_struct_refs,.false.)
  end function

  function gpufortrt_map_dec_struct_refs_c4_arr(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),dimension(*),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    call retval%init(c_loc(hostptr),0_c_size_t,gpufortrt_map_kind_dec_struct_refs,.false.)
  end function
                                                                
  function gpufortrt_map_dec_struct_refs_c8_scal(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    call retval%init(c_loc(hostptr),0_c_size_T,gpufortrt_map_kind_dec_struct_refs,.false.)
  end function

  function gpufortrt_map_dec_struct_refs_c8_arr(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),dimension(*),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    call retval%init(c_loc(hostptr),0_c_size_t,gpufortrt_map_kind_dec_struct_refs,.false.)
  end function

   ! gpufortrt_map_delete
  function gpufortrt_map_delete_b(hostptr,num_bytes) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    !
    type(c_ptr),intent(in)       :: hostptr
    integer(c_size_t),intent(in) :: num_bytes
    !
    type(mapping_t) :: retval
    !
    logical :: opt_declared_module_var
    !
    opt_declared_module_var = .false.
    call retval%init(hostptr,num_bytes,gpufortrt_map_kind_delete,opt_declared_module_var)
  end function

                                                                
  function gpufortrt_map_delete_l_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_l_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_l_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_l_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_l_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_l_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_l_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_l_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_delete_c_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_c_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_c_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_c_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_c_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_c_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_c_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_c_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_delete_i2_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_i2_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_i2_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_i2_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_i2_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_i2_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_i2_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_i2_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_delete_i4_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_i4_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_i4_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_i4_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_i4_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_i4_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_i4_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_i4_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_delete_i8_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(8,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_i8_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_i8_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_i8_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_i8_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_i8_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_i8_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_i8_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_delete_r4_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_r4_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_r4_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_r4_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_r4_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_r4_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_r4_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_r4_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_delete_r8_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(8,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_r8_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_r8_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_r8_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_r8_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_r8_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_r8_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_r8_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_delete_c4_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_c4_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_c4_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_c4_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_c4_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_c4_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_c4_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_c4_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_delete_c8_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_c8_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_c8_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_c8_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_c8_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_c8_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_c8_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_delete_c8_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

   
   
   ! gpufortrt_map_present
  function gpufortrt_map_present_b(hostptr,num_bytes) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    !
    type(c_ptr),intent(in)       :: hostptr
    integer(c_size_t),intent(in) :: num_bytes
    !
    type(mapping_t) :: retval
    !
    logical :: opt_declared_module_var
    !
    opt_declared_module_var = .false.
    call retval%init(hostptr,num_bytes,gpufortrt_map_kind_present,opt_declared_module_var)
  end function

                                                                
  function gpufortrt_map_present_l_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_l_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_l_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_l_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_l_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_l_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_l_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_l_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_present_c_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_c_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_c_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_c_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_c_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_c_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_c_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_c_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_present_i2_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_i2_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_i2_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_i2_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_i2_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_i2_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_i2_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_i2_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_present_i4_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_i4_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_i4_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_i4_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_i4_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_i4_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_i4_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_i4_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_present_i8_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(8,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_i8_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_i8_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_i8_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_i8_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_i8_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_i8_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_i8_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_present_r4_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_r4_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_r4_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_r4_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_r4_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_r4_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_r4_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_r4_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_present_r8_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(8,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_r8_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_r8_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_r8_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_r8_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_r8_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_r8_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_r8_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_present_c4_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_c4_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_c4_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_c4_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_c4_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_c4_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_c4_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_c4_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_present_c8_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_c8_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_c8_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_c8_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_c8_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_c8_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_c8_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_present_c8_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

   
   
   ! gpufortrt_map_no_create
  function gpufortrt_map_no_create_b(hostptr,num_bytes) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    !
    type(c_ptr),intent(in)       :: hostptr
    integer(c_size_t),intent(in) :: num_bytes
    !
    type(mapping_t) :: retval
    !
    logical :: opt_declared_module_var
    !
    opt_declared_module_var = .false.
    call retval%init(hostptr,num_bytes,gpufortrt_map_kind_no_create,opt_declared_module_var)
  end function

                                                                
  function gpufortrt_map_no_create_l_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_l_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_l_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_l_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_l_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_l_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_l_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_l_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_no_create_c_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_c_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_c_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_c_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_c_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_c_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_c_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_c_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_no_create_i2_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_i2_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_i2_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_i2_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_i2_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_i2_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_i2_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_i2_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_no_create_i4_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_i4_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_i4_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_i4_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_i4_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_i4_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_i4_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_i4_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_no_create_i8_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(8,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_i8_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_i8_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_i8_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_i8_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_i8_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_i8_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_i8_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_no_create_r4_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_r4_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_r4_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_r4_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_r4_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_r4_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_r4_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_r4_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_no_create_r8_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(8,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_r8_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_r8_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_r8_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_r8_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_r8_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_r8_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_r8_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_no_create_c4_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_c4_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_c4_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_c4_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_c4_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_c4_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_c4_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_c4_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_no_create_c8_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_c8_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_c8_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_c8_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_c8_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_c8_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_c8_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_no_create_c8_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

   
   
   ! gpufortrt_map_create
  function gpufortrt_map_create_b(hostptr,num_bytes,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    !
    type(c_ptr),intent(in)       :: hostptr
    integer(c_size_t),intent(in) :: num_bytes
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    logical :: opt_declared_module_var
    !
    opt_declared_module_var = .false.
    if ( present(declared_module_var) ) opt_declared_module_var = declared_module_var
    call retval%init(hostptr,num_bytes,gpufortrt_map_kind_create,opt_declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_l_0(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_l_1(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_l_2(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_l_3(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_l_4(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_l_5(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_l_6(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_l_7(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

   
                                                                
  function gpufortrt_map_create_c_0(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_c_1(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_c_2(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_c_3(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_c_4(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_c_5(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_c_6(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_c_7(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

   
                                                                
  function gpufortrt_map_create_i2_0(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_i2_1(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_i2_2(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_i2_3(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_i2_4(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_i2_5(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_i2_6(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_i2_7(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t),declared_module_var)
  end function

   
                                                                
  function gpufortrt_map_create_i4_0(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_i4_1(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_i4_2(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_i4_3(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_i4_4(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_i4_5(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_i4_6(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_i4_7(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

   
                                                                
  function gpufortrt_map_create_i8_0(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_i8_1(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_i8_2(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_i8_3(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_i8_4(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_i8_5(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_i8_6(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_i8_7(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

   
                                                                
  function gpufortrt_map_create_r4_0(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_r4_1(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_r4_2(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_r4_3(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_r4_4(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_r4_5(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_r4_6(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_r4_7(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

   
                                                                
  function gpufortrt_map_create_r8_0(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_r8_1(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_r8_2(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_r8_3(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_r8_4(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_r8_5(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_r8_6(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_r8_7(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

   
                                                                
  function gpufortrt_map_create_c4_0(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_c4_1(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_c4_2(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_c4_3(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_c4_4(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_c4_5(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_c4_6(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_c4_7(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t),declared_module_var)
  end function

   
                                                                
  function gpufortrt_map_create_c8_0(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_c8_1(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_c8_2(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_c8_3(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_c8_4(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_c8_5(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_c8_6(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_create_c8_7(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t),declared_module_var)
  end function

   
   
   ! gpufortrt_map_copy
  function gpufortrt_map_copy_b(hostptr,num_bytes) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    !
    type(c_ptr),intent(in)       :: hostptr
    integer(c_size_t),intent(in) :: num_bytes
    !
    type(mapping_t) :: retval
    !
    logical :: opt_declared_module_var
    !
    opt_declared_module_var = .false.
    call retval%init(hostptr,num_bytes,gpufortrt_map_kind_copy,opt_declared_module_var)
  end function

                                                                
  function gpufortrt_map_copy_l_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_l_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_l_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_l_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_l_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_l_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_l_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_l_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_copy_c_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_c_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_c_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_c_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_c_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_c_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_c_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_c_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_copy_i2_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_i2_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_i2_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_i2_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_i2_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_i2_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_i2_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_i2_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_copy_i4_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_i4_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_i4_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_i4_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_i4_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_i4_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_i4_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_i4_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_copy_i8_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_i8_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_i8_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_i8_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_i8_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_i8_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_i8_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_i8_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_copy_r4_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_r4_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_r4_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_r4_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_r4_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_r4_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_r4_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_r4_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_copy_r8_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_r8_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_r8_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_r8_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_r8_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_r8_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_r8_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_r8_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_copy_c4_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_c4_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_c4_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_c4_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_c4_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_c4_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_c4_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_c4_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_copy_c8_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_c8_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_c8_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_c8_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_c8_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_c8_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_c8_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copy_c8_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

   
   
   ! gpufortrt_map_copyin
  function gpufortrt_map_copyin_b(hostptr,num_bytes,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    !
    type(c_ptr),intent(in)       :: hostptr
    integer(c_size_t),intent(in) :: num_bytes
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    logical :: opt_declared_module_var
    !
    opt_declared_module_var = .false.
    if ( present(declared_module_var) ) opt_declared_module_var = declared_module_var
    call retval%init(hostptr,num_bytes,gpufortrt_map_kind_copyin,opt_declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_l_0(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_l_1(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_l_2(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_l_3(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_l_4(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_l_5(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_l_6(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_l_7(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

   
                                                                
  function gpufortrt_map_copyin_c_0(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_c_1(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_c_2(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_c_3(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_c_4(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_c_5(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_c_6(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_c_7(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t),declared_module_var)
  end function

   
                                                                
  function gpufortrt_map_copyin_i2_0(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_i2_1(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_i2_2(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_i2_3(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_i2_4(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_i2_5(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_i2_6(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_i2_7(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t),declared_module_var)
  end function

   
                                                                
  function gpufortrt_map_copyin_i4_0(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_i4_1(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_i4_2(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_i4_3(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_i4_4(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_i4_5(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_i4_6(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_i4_7(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

   
                                                                
  function gpufortrt_map_copyin_i8_0(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_i8_1(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_i8_2(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_i8_3(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_i8_4(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_i8_5(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_i8_6(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_i8_7(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

   
                                                                
  function gpufortrt_map_copyin_r4_0(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_r4_1(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_r4_2(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_r4_3(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_r4_4(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_r4_5(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_r4_6(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_r4_7(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t),declared_module_var)
  end function

   
                                                                
  function gpufortrt_map_copyin_r8_0(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_r8_1(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_r8_2(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_r8_3(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_r8_4(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_r8_5(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_r8_6(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_r8_7(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t),declared_module_var)
  end function

   
                                                                
  function gpufortrt_map_copyin_c4_0(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_c4_1(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_c4_2(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_c4_3(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_c4_4(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_c4_5(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_c4_6(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_c4_7(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t),declared_module_var)
  end function

   
                                                                
  function gpufortrt_map_copyin_c8_0(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_c8_1(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_c8_2(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_c8_3(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_c8_4(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_c8_5(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_c8_6(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t),declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyin_c8_7(hostptr,declared_module_var) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
     logical,intent(in),optional :: declared_module_var
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t),declared_module_var)
  end function

   
   
   ! gpufortrt_map_copyout
  function gpufortrt_map_copyout_b(hostptr,num_bytes) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    !
    type(c_ptr),intent(in)       :: hostptr
    integer(c_size_t),intent(in) :: num_bytes
    !
    type(mapping_t) :: retval
    !
    logical :: opt_declared_module_var
    !
    opt_declared_module_var = .false.
    call retval%init(hostptr,num_bytes,gpufortrt_map_kind_copyout,opt_declared_module_var)
  end function

                                                                
  function gpufortrt_map_copyout_l_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_l_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_l_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_l_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_l_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_l_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_l_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_l_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    logical,target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_copyout_c_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_c_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_c_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_c_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_c_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_c_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_c_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_c_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    character(c_char),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*1,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_copyout_i2_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_i2_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_i2_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_i2_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_i2_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_i2_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_i2_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_i2_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_short),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*2,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_copyout_i4_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_i4_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_i4_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_i4_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_i4_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_i4_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_i4_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_i4_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_int),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_copyout_i8_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_i8_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_i8_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_i8_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_i8_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_i8_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_i8_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_i8_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    integer(c_long),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_copyout_r4_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_r4_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_r4_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_r4_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_r4_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_r4_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_r4_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_r4_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_float),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*4,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_copyout_r8_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_r8_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_r8_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_r8_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_r8_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_r8_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_r8_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_r8_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    real(c_double),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*8,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_copyout_c4_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_c4_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_c4_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_c4_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_c4_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_c4_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_c4_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_c4_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_float_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*2*4,c_size_t))
  end function

   
                                                                
  function gpufortrt_map_copyout_c8_0(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_c8_1(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_c8_2(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_c8_3(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_c8_4(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_c8_5(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_c8_6(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

                                                                
  function gpufortrt_map_copyout_c8_7(hostptr) result(retval)
    use iso_c_binding
    use gpufortrt_core, only: mapping_t
    implicit none
    complex(c_double_complex),target,dimension(:,:,:,:,:,:,:),intent(in) :: hostptr
    !
    type(mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(size(hostptr)*2*8,c_size_t))
  end function

   
   

end module