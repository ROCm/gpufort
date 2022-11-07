! SPDX-License-Identifier: MIT
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
module gpufortrt_api
  use gpufortrt_api_core
  interface gpufortrt_map_present
    module procedure :: gpufortrt_map_present_b
    module procedure :: gpufortrt_map_present0_l1
    module procedure :: gpufortrt_map_present1_l1
    module procedure :: gpufortrt_map_present2_l1
    module procedure :: gpufortrt_map_present3_l1
    module procedure :: gpufortrt_map_present4_l1
    module procedure :: gpufortrt_map_present5_l1
    module procedure :: gpufortrt_map_present6_l1
    module procedure :: gpufortrt_map_present7_l1
    module procedure :: gpufortrt_map_present0_l4
    module procedure :: gpufortrt_map_present1_l4
    module procedure :: gpufortrt_map_present2_l4
    module procedure :: gpufortrt_map_present3_l4
    module procedure :: gpufortrt_map_present4_l4
    module procedure :: gpufortrt_map_present5_l4
    module procedure :: gpufortrt_map_present6_l4
    module procedure :: gpufortrt_map_present7_l4
    module procedure :: gpufortrt_map_present0_ch1
    module procedure :: gpufortrt_map_present1_ch1
    module procedure :: gpufortrt_map_present2_ch1
    module procedure :: gpufortrt_map_present3_ch1
    module procedure :: gpufortrt_map_present4_ch1
    module procedure :: gpufortrt_map_present5_ch1
    module procedure :: gpufortrt_map_present6_ch1
    module procedure :: gpufortrt_map_present7_ch1
    module procedure :: gpufortrt_map_present0_i1
    module procedure :: gpufortrt_map_present1_i1
    module procedure :: gpufortrt_map_present2_i1
    module procedure :: gpufortrt_map_present3_i1
    module procedure :: gpufortrt_map_present4_i1
    module procedure :: gpufortrt_map_present5_i1
    module procedure :: gpufortrt_map_present6_i1
    module procedure :: gpufortrt_map_present7_i1
    module procedure :: gpufortrt_map_present0_i2
    module procedure :: gpufortrt_map_present1_i2
    module procedure :: gpufortrt_map_present2_i2
    module procedure :: gpufortrt_map_present3_i2
    module procedure :: gpufortrt_map_present4_i2
    module procedure :: gpufortrt_map_present5_i2
    module procedure :: gpufortrt_map_present6_i2
    module procedure :: gpufortrt_map_present7_i2
    module procedure :: gpufortrt_map_present0_i4
    module procedure :: gpufortrt_map_present1_i4
    module procedure :: gpufortrt_map_present2_i4
    module procedure :: gpufortrt_map_present3_i4
    module procedure :: gpufortrt_map_present4_i4
    module procedure :: gpufortrt_map_present5_i4
    module procedure :: gpufortrt_map_present6_i4
    module procedure :: gpufortrt_map_present7_i4
    module procedure :: gpufortrt_map_present0_i8
    module procedure :: gpufortrt_map_present1_i8
    module procedure :: gpufortrt_map_present2_i8
    module procedure :: gpufortrt_map_present3_i8
    module procedure :: gpufortrt_map_present4_i8
    module procedure :: gpufortrt_map_present5_i8
    module procedure :: gpufortrt_map_present6_i8
    module procedure :: gpufortrt_map_present7_i8
    module procedure :: gpufortrt_map_present0_r4
    module procedure :: gpufortrt_map_present1_r4
    module procedure :: gpufortrt_map_present2_r4
    module procedure :: gpufortrt_map_present3_r4
    module procedure :: gpufortrt_map_present4_r4
    module procedure :: gpufortrt_map_present5_r4
    module procedure :: gpufortrt_map_present6_r4
    module procedure :: gpufortrt_map_present7_r4
    module procedure :: gpufortrt_map_present0_r8
    module procedure :: gpufortrt_map_present1_r8
    module procedure :: gpufortrt_map_present2_r8
    module procedure :: gpufortrt_map_present3_r8
    module procedure :: gpufortrt_map_present4_r8
    module procedure :: gpufortrt_map_present5_r8
    module procedure :: gpufortrt_map_present6_r8
    module procedure :: gpufortrt_map_present7_r8
    module procedure :: gpufortrt_map_present0_c4
    module procedure :: gpufortrt_map_present1_c4
    module procedure :: gpufortrt_map_present2_c4
    module procedure :: gpufortrt_map_present3_c4
    module procedure :: gpufortrt_map_present4_c4
    module procedure :: gpufortrt_map_present5_c4
    module procedure :: gpufortrt_map_present6_c4
    module procedure :: gpufortrt_map_present7_c4
    module procedure :: gpufortrt_map_present0_c8
    module procedure :: gpufortrt_map_present1_c8
    module procedure :: gpufortrt_map_present2_c8
    module procedure :: gpufortrt_map_present3_c8
    module procedure :: gpufortrt_map_present4_c8
    module procedure :: gpufortrt_map_present5_c8
    module procedure :: gpufortrt_map_present6_c8
    module procedure :: gpufortrt_map_present7_c8
  end interface

  interface gpufortrt_map_no_create
    module procedure :: gpufortrt_map_no_create_b
    module procedure :: gpufortrt_map_no_create0_l1
    module procedure :: gpufortrt_map_no_create1_l1
    module procedure :: gpufortrt_map_no_create2_l1
    module procedure :: gpufortrt_map_no_create3_l1
    module procedure :: gpufortrt_map_no_create4_l1
    module procedure :: gpufortrt_map_no_create5_l1
    module procedure :: gpufortrt_map_no_create6_l1
    module procedure :: gpufortrt_map_no_create7_l1
    module procedure :: gpufortrt_map_no_create0_l4
    module procedure :: gpufortrt_map_no_create1_l4
    module procedure :: gpufortrt_map_no_create2_l4
    module procedure :: gpufortrt_map_no_create3_l4
    module procedure :: gpufortrt_map_no_create4_l4
    module procedure :: gpufortrt_map_no_create5_l4
    module procedure :: gpufortrt_map_no_create6_l4
    module procedure :: gpufortrt_map_no_create7_l4
    module procedure :: gpufortrt_map_no_create0_ch1
    module procedure :: gpufortrt_map_no_create1_ch1
    module procedure :: gpufortrt_map_no_create2_ch1
    module procedure :: gpufortrt_map_no_create3_ch1
    module procedure :: gpufortrt_map_no_create4_ch1
    module procedure :: gpufortrt_map_no_create5_ch1
    module procedure :: gpufortrt_map_no_create6_ch1
    module procedure :: gpufortrt_map_no_create7_ch1
    module procedure :: gpufortrt_map_no_create0_i1
    module procedure :: gpufortrt_map_no_create1_i1
    module procedure :: gpufortrt_map_no_create2_i1
    module procedure :: gpufortrt_map_no_create3_i1
    module procedure :: gpufortrt_map_no_create4_i1
    module procedure :: gpufortrt_map_no_create5_i1
    module procedure :: gpufortrt_map_no_create6_i1
    module procedure :: gpufortrt_map_no_create7_i1
    module procedure :: gpufortrt_map_no_create0_i2
    module procedure :: gpufortrt_map_no_create1_i2
    module procedure :: gpufortrt_map_no_create2_i2
    module procedure :: gpufortrt_map_no_create3_i2
    module procedure :: gpufortrt_map_no_create4_i2
    module procedure :: gpufortrt_map_no_create5_i2
    module procedure :: gpufortrt_map_no_create6_i2
    module procedure :: gpufortrt_map_no_create7_i2
    module procedure :: gpufortrt_map_no_create0_i4
    module procedure :: gpufortrt_map_no_create1_i4
    module procedure :: gpufortrt_map_no_create2_i4
    module procedure :: gpufortrt_map_no_create3_i4
    module procedure :: gpufortrt_map_no_create4_i4
    module procedure :: gpufortrt_map_no_create5_i4
    module procedure :: gpufortrt_map_no_create6_i4
    module procedure :: gpufortrt_map_no_create7_i4
    module procedure :: gpufortrt_map_no_create0_i8
    module procedure :: gpufortrt_map_no_create1_i8
    module procedure :: gpufortrt_map_no_create2_i8
    module procedure :: gpufortrt_map_no_create3_i8
    module procedure :: gpufortrt_map_no_create4_i8
    module procedure :: gpufortrt_map_no_create5_i8
    module procedure :: gpufortrt_map_no_create6_i8
    module procedure :: gpufortrt_map_no_create7_i8
    module procedure :: gpufortrt_map_no_create0_r4
    module procedure :: gpufortrt_map_no_create1_r4
    module procedure :: gpufortrt_map_no_create2_r4
    module procedure :: gpufortrt_map_no_create3_r4
    module procedure :: gpufortrt_map_no_create4_r4
    module procedure :: gpufortrt_map_no_create5_r4
    module procedure :: gpufortrt_map_no_create6_r4
    module procedure :: gpufortrt_map_no_create7_r4
    module procedure :: gpufortrt_map_no_create0_r8
    module procedure :: gpufortrt_map_no_create1_r8
    module procedure :: gpufortrt_map_no_create2_r8
    module procedure :: gpufortrt_map_no_create3_r8
    module procedure :: gpufortrt_map_no_create4_r8
    module procedure :: gpufortrt_map_no_create5_r8
    module procedure :: gpufortrt_map_no_create6_r8
    module procedure :: gpufortrt_map_no_create7_r8
    module procedure :: gpufortrt_map_no_create0_c4
    module procedure :: gpufortrt_map_no_create1_c4
    module procedure :: gpufortrt_map_no_create2_c4
    module procedure :: gpufortrt_map_no_create3_c4
    module procedure :: gpufortrt_map_no_create4_c4
    module procedure :: gpufortrt_map_no_create5_c4
    module procedure :: gpufortrt_map_no_create6_c4
    module procedure :: gpufortrt_map_no_create7_c4
    module procedure :: gpufortrt_map_no_create0_c8
    module procedure :: gpufortrt_map_no_create1_c8
    module procedure :: gpufortrt_map_no_create2_c8
    module procedure :: gpufortrt_map_no_create3_c8
    module procedure :: gpufortrt_map_no_create4_c8
    module procedure :: gpufortrt_map_no_create5_c8
    module procedure :: gpufortrt_map_no_create6_c8
    module procedure :: gpufortrt_map_no_create7_c8
  end interface

  interface gpufortrt_map_create
    module procedure :: gpufortrt_map_create_b
    module procedure :: gpufortrt_map_create0_l1
    module procedure :: gpufortrt_map_create1_l1
    module procedure :: gpufortrt_map_create2_l1
    module procedure :: gpufortrt_map_create3_l1
    module procedure :: gpufortrt_map_create4_l1
    module procedure :: gpufortrt_map_create5_l1
    module procedure :: gpufortrt_map_create6_l1
    module procedure :: gpufortrt_map_create7_l1
    module procedure :: gpufortrt_map_create0_l4
    module procedure :: gpufortrt_map_create1_l4
    module procedure :: gpufortrt_map_create2_l4
    module procedure :: gpufortrt_map_create3_l4
    module procedure :: gpufortrt_map_create4_l4
    module procedure :: gpufortrt_map_create5_l4
    module procedure :: gpufortrt_map_create6_l4
    module procedure :: gpufortrt_map_create7_l4
    module procedure :: gpufortrt_map_create0_ch1
    module procedure :: gpufortrt_map_create1_ch1
    module procedure :: gpufortrt_map_create2_ch1
    module procedure :: gpufortrt_map_create3_ch1
    module procedure :: gpufortrt_map_create4_ch1
    module procedure :: gpufortrt_map_create5_ch1
    module procedure :: gpufortrt_map_create6_ch1
    module procedure :: gpufortrt_map_create7_ch1
    module procedure :: gpufortrt_map_create0_i1
    module procedure :: gpufortrt_map_create1_i1
    module procedure :: gpufortrt_map_create2_i1
    module procedure :: gpufortrt_map_create3_i1
    module procedure :: gpufortrt_map_create4_i1
    module procedure :: gpufortrt_map_create5_i1
    module procedure :: gpufortrt_map_create6_i1
    module procedure :: gpufortrt_map_create7_i1
    module procedure :: gpufortrt_map_create0_i2
    module procedure :: gpufortrt_map_create1_i2
    module procedure :: gpufortrt_map_create2_i2
    module procedure :: gpufortrt_map_create3_i2
    module procedure :: gpufortrt_map_create4_i2
    module procedure :: gpufortrt_map_create5_i2
    module procedure :: gpufortrt_map_create6_i2
    module procedure :: gpufortrt_map_create7_i2
    module procedure :: gpufortrt_map_create0_i4
    module procedure :: gpufortrt_map_create1_i4
    module procedure :: gpufortrt_map_create2_i4
    module procedure :: gpufortrt_map_create3_i4
    module procedure :: gpufortrt_map_create4_i4
    module procedure :: gpufortrt_map_create5_i4
    module procedure :: gpufortrt_map_create6_i4
    module procedure :: gpufortrt_map_create7_i4
    module procedure :: gpufortrt_map_create0_i8
    module procedure :: gpufortrt_map_create1_i8
    module procedure :: gpufortrt_map_create2_i8
    module procedure :: gpufortrt_map_create3_i8
    module procedure :: gpufortrt_map_create4_i8
    module procedure :: gpufortrt_map_create5_i8
    module procedure :: gpufortrt_map_create6_i8
    module procedure :: gpufortrt_map_create7_i8
    module procedure :: gpufortrt_map_create0_r4
    module procedure :: gpufortrt_map_create1_r4
    module procedure :: gpufortrt_map_create2_r4
    module procedure :: gpufortrt_map_create3_r4
    module procedure :: gpufortrt_map_create4_r4
    module procedure :: gpufortrt_map_create5_r4
    module procedure :: gpufortrt_map_create6_r4
    module procedure :: gpufortrt_map_create7_r4
    module procedure :: gpufortrt_map_create0_r8
    module procedure :: gpufortrt_map_create1_r8
    module procedure :: gpufortrt_map_create2_r8
    module procedure :: gpufortrt_map_create3_r8
    module procedure :: gpufortrt_map_create4_r8
    module procedure :: gpufortrt_map_create5_r8
    module procedure :: gpufortrt_map_create6_r8
    module procedure :: gpufortrt_map_create7_r8
    module procedure :: gpufortrt_map_create0_c4
    module procedure :: gpufortrt_map_create1_c4
    module procedure :: gpufortrt_map_create2_c4
    module procedure :: gpufortrt_map_create3_c4
    module procedure :: gpufortrt_map_create4_c4
    module procedure :: gpufortrt_map_create5_c4
    module procedure :: gpufortrt_map_create6_c4
    module procedure :: gpufortrt_map_create7_c4
    module procedure :: gpufortrt_map_create0_c8
    module procedure :: gpufortrt_map_create1_c8
    module procedure :: gpufortrt_map_create2_c8
    module procedure :: gpufortrt_map_create3_c8
    module procedure :: gpufortrt_map_create4_c8
    module procedure :: gpufortrt_map_create5_c8
    module procedure :: gpufortrt_map_create6_c8
    module procedure :: gpufortrt_map_create7_c8
  end interface

  interface gpufortrt_map_copyin
    module procedure :: gpufortrt_map_copyin_b
    module procedure :: gpufortrt_map_copyin0_l1
    module procedure :: gpufortrt_map_copyin1_l1
    module procedure :: gpufortrt_map_copyin2_l1
    module procedure :: gpufortrt_map_copyin3_l1
    module procedure :: gpufortrt_map_copyin4_l1
    module procedure :: gpufortrt_map_copyin5_l1
    module procedure :: gpufortrt_map_copyin6_l1
    module procedure :: gpufortrt_map_copyin7_l1
    module procedure :: gpufortrt_map_copyin0_l4
    module procedure :: gpufortrt_map_copyin1_l4
    module procedure :: gpufortrt_map_copyin2_l4
    module procedure :: gpufortrt_map_copyin3_l4
    module procedure :: gpufortrt_map_copyin4_l4
    module procedure :: gpufortrt_map_copyin5_l4
    module procedure :: gpufortrt_map_copyin6_l4
    module procedure :: gpufortrt_map_copyin7_l4
    module procedure :: gpufortrt_map_copyin0_ch1
    module procedure :: gpufortrt_map_copyin1_ch1
    module procedure :: gpufortrt_map_copyin2_ch1
    module procedure :: gpufortrt_map_copyin3_ch1
    module procedure :: gpufortrt_map_copyin4_ch1
    module procedure :: gpufortrt_map_copyin5_ch1
    module procedure :: gpufortrt_map_copyin6_ch1
    module procedure :: gpufortrt_map_copyin7_ch1
    module procedure :: gpufortrt_map_copyin0_i1
    module procedure :: gpufortrt_map_copyin1_i1
    module procedure :: gpufortrt_map_copyin2_i1
    module procedure :: gpufortrt_map_copyin3_i1
    module procedure :: gpufortrt_map_copyin4_i1
    module procedure :: gpufortrt_map_copyin5_i1
    module procedure :: gpufortrt_map_copyin6_i1
    module procedure :: gpufortrt_map_copyin7_i1
    module procedure :: gpufortrt_map_copyin0_i2
    module procedure :: gpufortrt_map_copyin1_i2
    module procedure :: gpufortrt_map_copyin2_i2
    module procedure :: gpufortrt_map_copyin3_i2
    module procedure :: gpufortrt_map_copyin4_i2
    module procedure :: gpufortrt_map_copyin5_i2
    module procedure :: gpufortrt_map_copyin6_i2
    module procedure :: gpufortrt_map_copyin7_i2
    module procedure :: gpufortrt_map_copyin0_i4
    module procedure :: gpufortrt_map_copyin1_i4
    module procedure :: gpufortrt_map_copyin2_i4
    module procedure :: gpufortrt_map_copyin3_i4
    module procedure :: gpufortrt_map_copyin4_i4
    module procedure :: gpufortrt_map_copyin5_i4
    module procedure :: gpufortrt_map_copyin6_i4
    module procedure :: gpufortrt_map_copyin7_i4
    module procedure :: gpufortrt_map_copyin0_i8
    module procedure :: gpufortrt_map_copyin1_i8
    module procedure :: gpufortrt_map_copyin2_i8
    module procedure :: gpufortrt_map_copyin3_i8
    module procedure :: gpufortrt_map_copyin4_i8
    module procedure :: gpufortrt_map_copyin5_i8
    module procedure :: gpufortrt_map_copyin6_i8
    module procedure :: gpufortrt_map_copyin7_i8
    module procedure :: gpufortrt_map_copyin0_r4
    module procedure :: gpufortrt_map_copyin1_r4
    module procedure :: gpufortrt_map_copyin2_r4
    module procedure :: gpufortrt_map_copyin3_r4
    module procedure :: gpufortrt_map_copyin4_r4
    module procedure :: gpufortrt_map_copyin5_r4
    module procedure :: gpufortrt_map_copyin6_r4
    module procedure :: gpufortrt_map_copyin7_r4
    module procedure :: gpufortrt_map_copyin0_r8
    module procedure :: gpufortrt_map_copyin1_r8
    module procedure :: gpufortrt_map_copyin2_r8
    module procedure :: gpufortrt_map_copyin3_r8
    module procedure :: gpufortrt_map_copyin4_r8
    module procedure :: gpufortrt_map_copyin5_r8
    module procedure :: gpufortrt_map_copyin6_r8
    module procedure :: gpufortrt_map_copyin7_r8
    module procedure :: gpufortrt_map_copyin0_c4
    module procedure :: gpufortrt_map_copyin1_c4
    module procedure :: gpufortrt_map_copyin2_c4
    module procedure :: gpufortrt_map_copyin3_c4
    module procedure :: gpufortrt_map_copyin4_c4
    module procedure :: gpufortrt_map_copyin5_c4
    module procedure :: gpufortrt_map_copyin6_c4
    module procedure :: gpufortrt_map_copyin7_c4
    module procedure :: gpufortrt_map_copyin0_c8
    module procedure :: gpufortrt_map_copyin1_c8
    module procedure :: gpufortrt_map_copyin2_c8
    module procedure :: gpufortrt_map_copyin3_c8
    module procedure :: gpufortrt_map_copyin4_c8
    module procedure :: gpufortrt_map_copyin5_c8
    module procedure :: gpufortrt_map_copyin6_c8
    module procedure :: gpufortrt_map_copyin7_c8
  end interface

  interface gpufortrt_map_copy
    module procedure :: gpufortrt_map_copy_b
    module procedure :: gpufortrt_map_copy0_l1
    module procedure :: gpufortrt_map_copy1_l1
    module procedure :: gpufortrt_map_copy2_l1
    module procedure :: gpufortrt_map_copy3_l1
    module procedure :: gpufortrt_map_copy4_l1
    module procedure :: gpufortrt_map_copy5_l1
    module procedure :: gpufortrt_map_copy6_l1
    module procedure :: gpufortrt_map_copy7_l1
    module procedure :: gpufortrt_map_copy0_l4
    module procedure :: gpufortrt_map_copy1_l4
    module procedure :: gpufortrt_map_copy2_l4
    module procedure :: gpufortrt_map_copy3_l4
    module procedure :: gpufortrt_map_copy4_l4
    module procedure :: gpufortrt_map_copy5_l4
    module procedure :: gpufortrt_map_copy6_l4
    module procedure :: gpufortrt_map_copy7_l4
    module procedure :: gpufortrt_map_copy0_ch1
    module procedure :: gpufortrt_map_copy1_ch1
    module procedure :: gpufortrt_map_copy2_ch1
    module procedure :: gpufortrt_map_copy3_ch1
    module procedure :: gpufortrt_map_copy4_ch1
    module procedure :: gpufortrt_map_copy5_ch1
    module procedure :: gpufortrt_map_copy6_ch1
    module procedure :: gpufortrt_map_copy7_ch1
    module procedure :: gpufortrt_map_copy0_i1
    module procedure :: gpufortrt_map_copy1_i1
    module procedure :: gpufortrt_map_copy2_i1
    module procedure :: gpufortrt_map_copy3_i1
    module procedure :: gpufortrt_map_copy4_i1
    module procedure :: gpufortrt_map_copy5_i1
    module procedure :: gpufortrt_map_copy6_i1
    module procedure :: gpufortrt_map_copy7_i1
    module procedure :: gpufortrt_map_copy0_i2
    module procedure :: gpufortrt_map_copy1_i2
    module procedure :: gpufortrt_map_copy2_i2
    module procedure :: gpufortrt_map_copy3_i2
    module procedure :: gpufortrt_map_copy4_i2
    module procedure :: gpufortrt_map_copy5_i2
    module procedure :: gpufortrt_map_copy6_i2
    module procedure :: gpufortrt_map_copy7_i2
    module procedure :: gpufortrt_map_copy0_i4
    module procedure :: gpufortrt_map_copy1_i4
    module procedure :: gpufortrt_map_copy2_i4
    module procedure :: gpufortrt_map_copy3_i4
    module procedure :: gpufortrt_map_copy4_i4
    module procedure :: gpufortrt_map_copy5_i4
    module procedure :: gpufortrt_map_copy6_i4
    module procedure :: gpufortrt_map_copy7_i4
    module procedure :: gpufortrt_map_copy0_i8
    module procedure :: gpufortrt_map_copy1_i8
    module procedure :: gpufortrt_map_copy2_i8
    module procedure :: gpufortrt_map_copy3_i8
    module procedure :: gpufortrt_map_copy4_i8
    module procedure :: gpufortrt_map_copy5_i8
    module procedure :: gpufortrt_map_copy6_i8
    module procedure :: gpufortrt_map_copy7_i8
    module procedure :: gpufortrt_map_copy0_r4
    module procedure :: gpufortrt_map_copy1_r4
    module procedure :: gpufortrt_map_copy2_r4
    module procedure :: gpufortrt_map_copy3_r4
    module procedure :: gpufortrt_map_copy4_r4
    module procedure :: gpufortrt_map_copy5_r4
    module procedure :: gpufortrt_map_copy6_r4
    module procedure :: gpufortrt_map_copy7_r4
    module procedure :: gpufortrt_map_copy0_r8
    module procedure :: gpufortrt_map_copy1_r8
    module procedure :: gpufortrt_map_copy2_r8
    module procedure :: gpufortrt_map_copy3_r8
    module procedure :: gpufortrt_map_copy4_r8
    module procedure :: gpufortrt_map_copy5_r8
    module procedure :: gpufortrt_map_copy6_r8
    module procedure :: gpufortrt_map_copy7_r8
    module procedure :: gpufortrt_map_copy0_c4
    module procedure :: gpufortrt_map_copy1_c4
    module procedure :: gpufortrt_map_copy2_c4
    module procedure :: gpufortrt_map_copy3_c4
    module procedure :: gpufortrt_map_copy4_c4
    module procedure :: gpufortrt_map_copy5_c4
    module procedure :: gpufortrt_map_copy6_c4
    module procedure :: gpufortrt_map_copy7_c4
    module procedure :: gpufortrt_map_copy0_c8
    module procedure :: gpufortrt_map_copy1_c8
    module procedure :: gpufortrt_map_copy2_c8
    module procedure :: gpufortrt_map_copy3_c8
    module procedure :: gpufortrt_map_copy4_c8
    module procedure :: gpufortrt_map_copy5_c8
    module procedure :: gpufortrt_map_copy6_c8
    module procedure :: gpufortrt_map_copy7_c8
  end interface

  interface gpufortrt_map_copyout
    module procedure :: gpufortrt_map_copyout_b
    module procedure :: gpufortrt_map_copyout0_l1
    module procedure :: gpufortrt_map_copyout1_l1
    module procedure :: gpufortrt_map_copyout2_l1
    module procedure :: gpufortrt_map_copyout3_l1
    module procedure :: gpufortrt_map_copyout4_l1
    module procedure :: gpufortrt_map_copyout5_l1
    module procedure :: gpufortrt_map_copyout6_l1
    module procedure :: gpufortrt_map_copyout7_l1
    module procedure :: gpufortrt_map_copyout0_l4
    module procedure :: gpufortrt_map_copyout1_l4
    module procedure :: gpufortrt_map_copyout2_l4
    module procedure :: gpufortrt_map_copyout3_l4
    module procedure :: gpufortrt_map_copyout4_l4
    module procedure :: gpufortrt_map_copyout5_l4
    module procedure :: gpufortrt_map_copyout6_l4
    module procedure :: gpufortrt_map_copyout7_l4
    module procedure :: gpufortrt_map_copyout0_ch1
    module procedure :: gpufortrt_map_copyout1_ch1
    module procedure :: gpufortrt_map_copyout2_ch1
    module procedure :: gpufortrt_map_copyout3_ch1
    module procedure :: gpufortrt_map_copyout4_ch1
    module procedure :: gpufortrt_map_copyout5_ch1
    module procedure :: gpufortrt_map_copyout6_ch1
    module procedure :: gpufortrt_map_copyout7_ch1
    module procedure :: gpufortrt_map_copyout0_i1
    module procedure :: gpufortrt_map_copyout1_i1
    module procedure :: gpufortrt_map_copyout2_i1
    module procedure :: gpufortrt_map_copyout3_i1
    module procedure :: gpufortrt_map_copyout4_i1
    module procedure :: gpufortrt_map_copyout5_i1
    module procedure :: gpufortrt_map_copyout6_i1
    module procedure :: gpufortrt_map_copyout7_i1
    module procedure :: gpufortrt_map_copyout0_i2
    module procedure :: gpufortrt_map_copyout1_i2
    module procedure :: gpufortrt_map_copyout2_i2
    module procedure :: gpufortrt_map_copyout3_i2
    module procedure :: gpufortrt_map_copyout4_i2
    module procedure :: gpufortrt_map_copyout5_i2
    module procedure :: gpufortrt_map_copyout6_i2
    module procedure :: gpufortrt_map_copyout7_i2
    module procedure :: gpufortrt_map_copyout0_i4
    module procedure :: gpufortrt_map_copyout1_i4
    module procedure :: gpufortrt_map_copyout2_i4
    module procedure :: gpufortrt_map_copyout3_i4
    module procedure :: gpufortrt_map_copyout4_i4
    module procedure :: gpufortrt_map_copyout5_i4
    module procedure :: gpufortrt_map_copyout6_i4
    module procedure :: gpufortrt_map_copyout7_i4
    module procedure :: gpufortrt_map_copyout0_i8
    module procedure :: gpufortrt_map_copyout1_i8
    module procedure :: gpufortrt_map_copyout2_i8
    module procedure :: gpufortrt_map_copyout3_i8
    module procedure :: gpufortrt_map_copyout4_i8
    module procedure :: gpufortrt_map_copyout5_i8
    module procedure :: gpufortrt_map_copyout6_i8
    module procedure :: gpufortrt_map_copyout7_i8
    module procedure :: gpufortrt_map_copyout0_r4
    module procedure :: gpufortrt_map_copyout1_r4
    module procedure :: gpufortrt_map_copyout2_r4
    module procedure :: gpufortrt_map_copyout3_r4
    module procedure :: gpufortrt_map_copyout4_r4
    module procedure :: gpufortrt_map_copyout5_r4
    module procedure :: gpufortrt_map_copyout6_r4
    module procedure :: gpufortrt_map_copyout7_r4
    module procedure :: gpufortrt_map_copyout0_r8
    module procedure :: gpufortrt_map_copyout1_r8
    module procedure :: gpufortrt_map_copyout2_r8
    module procedure :: gpufortrt_map_copyout3_r8
    module procedure :: gpufortrt_map_copyout4_r8
    module procedure :: gpufortrt_map_copyout5_r8
    module procedure :: gpufortrt_map_copyout6_r8
    module procedure :: gpufortrt_map_copyout7_r8
    module procedure :: gpufortrt_map_copyout0_c4
    module procedure :: gpufortrt_map_copyout1_c4
    module procedure :: gpufortrt_map_copyout2_c4
    module procedure :: gpufortrt_map_copyout3_c4
    module procedure :: gpufortrt_map_copyout4_c4
    module procedure :: gpufortrt_map_copyout5_c4
    module procedure :: gpufortrt_map_copyout6_c4
    module procedure :: gpufortrt_map_copyout7_c4
    module procedure :: gpufortrt_map_copyout0_c8
    module procedure :: gpufortrt_map_copyout1_c8
    module procedure :: gpufortrt_map_copyout2_c8
    module procedure :: gpufortrt_map_copyout3_c8
    module procedure :: gpufortrt_map_copyout4_c8
    module procedure :: gpufortrt_map_copyout5_c8
    module procedure :: gpufortrt_map_copyout6_c8
    module procedure :: gpufortrt_map_copyout7_c8
  end interface

  interface gpufortrt_map_delete
    module procedure :: gpufortrt_map_delete_b
    module procedure :: gpufortrt_map_delete0_l1
    module procedure :: gpufortrt_map_delete1_l1
    module procedure :: gpufortrt_map_delete2_l1
    module procedure :: gpufortrt_map_delete3_l1
    module procedure :: gpufortrt_map_delete4_l1
    module procedure :: gpufortrt_map_delete5_l1
    module procedure :: gpufortrt_map_delete6_l1
    module procedure :: gpufortrt_map_delete7_l1
    module procedure :: gpufortrt_map_delete0_l4
    module procedure :: gpufortrt_map_delete1_l4
    module procedure :: gpufortrt_map_delete2_l4
    module procedure :: gpufortrt_map_delete3_l4
    module procedure :: gpufortrt_map_delete4_l4
    module procedure :: gpufortrt_map_delete5_l4
    module procedure :: gpufortrt_map_delete6_l4
    module procedure :: gpufortrt_map_delete7_l4
    module procedure :: gpufortrt_map_delete0_ch1
    module procedure :: gpufortrt_map_delete1_ch1
    module procedure :: gpufortrt_map_delete2_ch1
    module procedure :: gpufortrt_map_delete3_ch1
    module procedure :: gpufortrt_map_delete4_ch1
    module procedure :: gpufortrt_map_delete5_ch1
    module procedure :: gpufortrt_map_delete6_ch1
    module procedure :: gpufortrt_map_delete7_ch1
    module procedure :: gpufortrt_map_delete0_i1
    module procedure :: gpufortrt_map_delete1_i1
    module procedure :: gpufortrt_map_delete2_i1
    module procedure :: gpufortrt_map_delete3_i1
    module procedure :: gpufortrt_map_delete4_i1
    module procedure :: gpufortrt_map_delete5_i1
    module procedure :: gpufortrt_map_delete6_i1
    module procedure :: gpufortrt_map_delete7_i1
    module procedure :: gpufortrt_map_delete0_i2
    module procedure :: gpufortrt_map_delete1_i2
    module procedure :: gpufortrt_map_delete2_i2
    module procedure :: gpufortrt_map_delete3_i2
    module procedure :: gpufortrt_map_delete4_i2
    module procedure :: gpufortrt_map_delete5_i2
    module procedure :: gpufortrt_map_delete6_i2
    module procedure :: gpufortrt_map_delete7_i2
    module procedure :: gpufortrt_map_delete0_i4
    module procedure :: gpufortrt_map_delete1_i4
    module procedure :: gpufortrt_map_delete2_i4
    module procedure :: gpufortrt_map_delete3_i4
    module procedure :: gpufortrt_map_delete4_i4
    module procedure :: gpufortrt_map_delete5_i4
    module procedure :: gpufortrt_map_delete6_i4
    module procedure :: gpufortrt_map_delete7_i4
    module procedure :: gpufortrt_map_delete0_i8
    module procedure :: gpufortrt_map_delete1_i8
    module procedure :: gpufortrt_map_delete2_i8
    module procedure :: gpufortrt_map_delete3_i8
    module procedure :: gpufortrt_map_delete4_i8
    module procedure :: gpufortrt_map_delete5_i8
    module procedure :: gpufortrt_map_delete6_i8
    module procedure :: gpufortrt_map_delete7_i8
    module procedure :: gpufortrt_map_delete0_r4
    module procedure :: gpufortrt_map_delete1_r4
    module procedure :: gpufortrt_map_delete2_r4
    module procedure :: gpufortrt_map_delete3_r4
    module procedure :: gpufortrt_map_delete4_r4
    module procedure :: gpufortrt_map_delete5_r4
    module procedure :: gpufortrt_map_delete6_r4
    module procedure :: gpufortrt_map_delete7_r4
    module procedure :: gpufortrt_map_delete0_r8
    module procedure :: gpufortrt_map_delete1_r8
    module procedure :: gpufortrt_map_delete2_r8
    module procedure :: gpufortrt_map_delete3_r8
    module procedure :: gpufortrt_map_delete4_r8
    module procedure :: gpufortrt_map_delete5_r8
    module procedure :: gpufortrt_map_delete6_r8
    module procedure :: gpufortrt_map_delete7_r8
    module procedure :: gpufortrt_map_delete0_c4
    module procedure :: gpufortrt_map_delete1_c4
    module procedure :: gpufortrt_map_delete2_c4
    module procedure :: gpufortrt_map_delete3_c4
    module procedure :: gpufortrt_map_delete4_c4
    module procedure :: gpufortrt_map_delete5_c4
    module procedure :: gpufortrt_map_delete6_c4
    module procedure :: gpufortrt_map_delete7_c4
    module procedure :: gpufortrt_map_delete0_c8
    module procedure :: gpufortrt_map_delete1_c8
    module procedure :: gpufortrt_map_delete2_c8
    module procedure :: gpufortrt_map_delete3_c8
    module procedure :: gpufortrt_map_delete4_c8
    module procedure :: gpufortrt_map_delete5_c8
    module procedure :: gpufortrt_map_delete6_c8
    module procedure :: gpufortrt_map_delete7_c8
  end interface



  interface gpufortrt_present
    module procedure :: gpufortrt_present0_l1
    module procedure :: gpufortrt_present1_l1
    module procedure :: gpufortrt_present2_l1
    module procedure :: gpufortrt_present3_l1
    module procedure :: gpufortrt_present4_l1
    module procedure :: gpufortrt_present5_l1
    module procedure :: gpufortrt_present6_l1
    module procedure :: gpufortrt_present7_l1
    module procedure :: gpufortrt_present0_l4
    module procedure :: gpufortrt_present1_l4
    module procedure :: gpufortrt_present2_l4
    module procedure :: gpufortrt_present3_l4
    module procedure :: gpufortrt_present4_l4
    module procedure :: gpufortrt_present5_l4
    module procedure :: gpufortrt_present6_l4
    module procedure :: gpufortrt_present7_l4
    module procedure :: gpufortrt_present0_ch1
    module procedure :: gpufortrt_present1_ch1
    module procedure :: gpufortrt_present2_ch1
    module procedure :: gpufortrt_present3_ch1
    module procedure :: gpufortrt_present4_ch1
    module procedure :: gpufortrt_present5_ch1
    module procedure :: gpufortrt_present6_ch1
    module procedure :: gpufortrt_present7_ch1
    module procedure :: gpufortrt_present0_i1
    module procedure :: gpufortrt_present1_i1
    module procedure :: gpufortrt_present2_i1
    module procedure :: gpufortrt_present3_i1
    module procedure :: gpufortrt_present4_i1
    module procedure :: gpufortrt_present5_i1
    module procedure :: gpufortrt_present6_i1
    module procedure :: gpufortrt_present7_i1
    module procedure :: gpufortrt_present0_i2
    module procedure :: gpufortrt_present1_i2
    module procedure :: gpufortrt_present2_i2
    module procedure :: gpufortrt_present3_i2
    module procedure :: gpufortrt_present4_i2
    module procedure :: gpufortrt_present5_i2
    module procedure :: gpufortrt_present6_i2
    module procedure :: gpufortrt_present7_i2
    module procedure :: gpufortrt_present0_i4
    module procedure :: gpufortrt_present1_i4
    module procedure :: gpufortrt_present2_i4
    module procedure :: gpufortrt_present3_i4
    module procedure :: gpufortrt_present4_i4
    module procedure :: gpufortrt_present5_i4
    module procedure :: gpufortrt_present6_i4
    module procedure :: gpufortrt_present7_i4
    module procedure :: gpufortrt_present0_i8
    module procedure :: gpufortrt_present1_i8
    module procedure :: gpufortrt_present2_i8
    module procedure :: gpufortrt_present3_i8
    module procedure :: gpufortrt_present4_i8
    module procedure :: gpufortrt_present5_i8
    module procedure :: gpufortrt_present6_i8
    module procedure :: gpufortrt_present7_i8
    module procedure :: gpufortrt_present0_r4
    module procedure :: gpufortrt_present1_r4
    module procedure :: gpufortrt_present2_r4
    module procedure :: gpufortrt_present3_r4
    module procedure :: gpufortrt_present4_r4
    module procedure :: gpufortrt_present5_r4
    module procedure :: gpufortrt_present6_r4
    module procedure :: gpufortrt_present7_r4
    module procedure :: gpufortrt_present0_r8
    module procedure :: gpufortrt_present1_r8
    module procedure :: gpufortrt_present2_r8
    module procedure :: gpufortrt_present3_r8
    module procedure :: gpufortrt_present4_r8
    module procedure :: gpufortrt_present5_r8
    module procedure :: gpufortrt_present6_r8
    module procedure :: gpufortrt_present7_r8
    module procedure :: gpufortrt_present0_c4
    module procedure :: gpufortrt_present1_c4
    module procedure :: gpufortrt_present2_c4
    module procedure :: gpufortrt_present3_c4
    module procedure :: gpufortrt_present4_c4
    module procedure :: gpufortrt_present5_c4
    module procedure :: gpufortrt_present6_c4
    module procedure :: gpufortrt_present7_c4
    module procedure :: gpufortrt_present0_c8
    module procedure :: gpufortrt_present1_c8
    module procedure :: gpufortrt_present2_c8
    module procedure :: gpufortrt_present3_c8
    module procedure :: gpufortrt_present4_c8
    module procedure :: gpufortrt_present5_c8
    module procedure :: gpufortrt_present6_c8
    module procedure :: gpufortrt_present7_c8
  end interface


  interface gpufortrt_create
    module procedure :: gpufortrt_create_b
    module procedure :: gpufortrt_create0_l1
    module procedure :: gpufortrt_create1_l1
    module procedure :: gpufortrt_create2_l1
    module procedure :: gpufortrt_create3_l1
    module procedure :: gpufortrt_create4_l1
    module procedure :: gpufortrt_create5_l1
    module procedure :: gpufortrt_create6_l1
    module procedure :: gpufortrt_create7_l1
    module procedure :: gpufortrt_create0_l4
    module procedure :: gpufortrt_create1_l4
    module procedure :: gpufortrt_create2_l4
    module procedure :: gpufortrt_create3_l4
    module procedure :: gpufortrt_create4_l4
    module procedure :: gpufortrt_create5_l4
    module procedure :: gpufortrt_create6_l4
    module procedure :: gpufortrt_create7_l4
    module procedure :: gpufortrt_create0_ch1
    module procedure :: gpufortrt_create1_ch1
    module procedure :: gpufortrt_create2_ch1
    module procedure :: gpufortrt_create3_ch1
    module procedure :: gpufortrt_create4_ch1
    module procedure :: gpufortrt_create5_ch1
    module procedure :: gpufortrt_create6_ch1
    module procedure :: gpufortrt_create7_ch1
    module procedure :: gpufortrt_create0_i1
    module procedure :: gpufortrt_create1_i1
    module procedure :: gpufortrt_create2_i1
    module procedure :: gpufortrt_create3_i1
    module procedure :: gpufortrt_create4_i1
    module procedure :: gpufortrt_create5_i1
    module procedure :: gpufortrt_create6_i1
    module procedure :: gpufortrt_create7_i1
    module procedure :: gpufortrt_create0_i2
    module procedure :: gpufortrt_create1_i2
    module procedure :: gpufortrt_create2_i2
    module procedure :: gpufortrt_create3_i2
    module procedure :: gpufortrt_create4_i2
    module procedure :: gpufortrt_create5_i2
    module procedure :: gpufortrt_create6_i2
    module procedure :: gpufortrt_create7_i2
    module procedure :: gpufortrt_create0_i4
    module procedure :: gpufortrt_create1_i4
    module procedure :: gpufortrt_create2_i4
    module procedure :: gpufortrt_create3_i4
    module procedure :: gpufortrt_create4_i4
    module procedure :: gpufortrt_create5_i4
    module procedure :: gpufortrt_create6_i4
    module procedure :: gpufortrt_create7_i4
    module procedure :: gpufortrt_create0_i8
    module procedure :: gpufortrt_create1_i8
    module procedure :: gpufortrt_create2_i8
    module procedure :: gpufortrt_create3_i8
    module procedure :: gpufortrt_create4_i8
    module procedure :: gpufortrt_create5_i8
    module procedure :: gpufortrt_create6_i8
    module procedure :: gpufortrt_create7_i8
    module procedure :: gpufortrt_create0_r4
    module procedure :: gpufortrt_create1_r4
    module procedure :: gpufortrt_create2_r4
    module procedure :: gpufortrt_create3_r4
    module procedure :: gpufortrt_create4_r4
    module procedure :: gpufortrt_create5_r4
    module procedure :: gpufortrt_create6_r4
    module procedure :: gpufortrt_create7_r4
    module procedure :: gpufortrt_create0_r8
    module procedure :: gpufortrt_create1_r8
    module procedure :: gpufortrt_create2_r8
    module procedure :: gpufortrt_create3_r8
    module procedure :: gpufortrt_create4_r8
    module procedure :: gpufortrt_create5_r8
    module procedure :: gpufortrt_create6_r8
    module procedure :: gpufortrt_create7_r8
    module procedure :: gpufortrt_create0_c4
    module procedure :: gpufortrt_create1_c4
    module procedure :: gpufortrt_create2_c4
    module procedure :: gpufortrt_create3_c4
    module procedure :: gpufortrt_create4_c4
    module procedure :: gpufortrt_create5_c4
    module procedure :: gpufortrt_create6_c4
    module procedure :: gpufortrt_create7_c4
    module procedure :: gpufortrt_create0_c8
    module procedure :: gpufortrt_create1_c8
    module procedure :: gpufortrt_create2_c8
    module procedure :: gpufortrt_create3_c8
    module procedure :: gpufortrt_create4_c8
    module procedure :: gpufortrt_create5_c8
    module procedure :: gpufortrt_create6_c8
    module procedure :: gpufortrt_create7_c8
  end interface


  interface gpufortrt_copyin
    module procedure :: gpufortrt_copyin_b
    module procedure :: gpufortrt_copyin0_l1
    module procedure :: gpufortrt_copyin1_l1
    module procedure :: gpufortrt_copyin2_l1
    module procedure :: gpufortrt_copyin3_l1
    module procedure :: gpufortrt_copyin4_l1
    module procedure :: gpufortrt_copyin5_l1
    module procedure :: gpufortrt_copyin6_l1
    module procedure :: gpufortrt_copyin7_l1
    module procedure :: gpufortrt_copyin0_l4
    module procedure :: gpufortrt_copyin1_l4
    module procedure :: gpufortrt_copyin2_l4
    module procedure :: gpufortrt_copyin3_l4
    module procedure :: gpufortrt_copyin4_l4
    module procedure :: gpufortrt_copyin5_l4
    module procedure :: gpufortrt_copyin6_l4
    module procedure :: gpufortrt_copyin7_l4
    module procedure :: gpufortrt_copyin0_ch1
    module procedure :: gpufortrt_copyin1_ch1
    module procedure :: gpufortrt_copyin2_ch1
    module procedure :: gpufortrt_copyin3_ch1
    module procedure :: gpufortrt_copyin4_ch1
    module procedure :: gpufortrt_copyin5_ch1
    module procedure :: gpufortrt_copyin6_ch1
    module procedure :: gpufortrt_copyin7_ch1
    module procedure :: gpufortrt_copyin0_i1
    module procedure :: gpufortrt_copyin1_i1
    module procedure :: gpufortrt_copyin2_i1
    module procedure :: gpufortrt_copyin3_i1
    module procedure :: gpufortrt_copyin4_i1
    module procedure :: gpufortrt_copyin5_i1
    module procedure :: gpufortrt_copyin6_i1
    module procedure :: gpufortrt_copyin7_i1
    module procedure :: gpufortrt_copyin0_i2
    module procedure :: gpufortrt_copyin1_i2
    module procedure :: gpufortrt_copyin2_i2
    module procedure :: gpufortrt_copyin3_i2
    module procedure :: gpufortrt_copyin4_i2
    module procedure :: gpufortrt_copyin5_i2
    module procedure :: gpufortrt_copyin6_i2
    module procedure :: gpufortrt_copyin7_i2
    module procedure :: gpufortrt_copyin0_i4
    module procedure :: gpufortrt_copyin1_i4
    module procedure :: gpufortrt_copyin2_i4
    module procedure :: gpufortrt_copyin3_i4
    module procedure :: gpufortrt_copyin4_i4
    module procedure :: gpufortrt_copyin5_i4
    module procedure :: gpufortrt_copyin6_i4
    module procedure :: gpufortrt_copyin7_i4
    module procedure :: gpufortrt_copyin0_i8
    module procedure :: gpufortrt_copyin1_i8
    module procedure :: gpufortrt_copyin2_i8
    module procedure :: gpufortrt_copyin3_i8
    module procedure :: gpufortrt_copyin4_i8
    module procedure :: gpufortrt_copyin5_i8
    module procedure :: gpufortrt_copyin6_i8
    module procedure :: gpufortrt_copyin7_i8
    module procedure :: gpufortrt_copyin0_r4
    module procedure :: gpufortrt_copyin1_r4
    module procedure :: gpufortrt_copyin2_r4
    module procedure :: gpufortrt_copyin3_r4
    module procedure :: gpufortrt_copyin4_r4
    module procedure :: gpufortrt_copyin5_r4
    module procedure :: gpufortrt_copyin6_r4
    module procedure :: gpufortrt_copyin7_r4
    module procedure :: gpufortrt_copyin0_r8
    module procedure :: gpufortrt_copyin1_r8
    module procedure :: gpufortrt_copyin2_r8
    module procedure :: gpufortrt_copyin3_r8
    module procedure :: gpufortrt_copyin4_r8
    module procedure :: gpufortrt_copyin5_r8
    module procedure :: gpufortrt_copyin6_r8
    module procedure :: gpufortrt_copyin7_r8
    module procedure :: gpufortrt_copyin0_c4
    module procedure :: gpufortrt_copyin1_c4
    module procedure :: gpufortrt_copyin2_c4
    module procedure :: gpufortrt_copyin3_c4
    module procedure :: gpufortrt_copyin4_c4
    module procedure :: gpufortrt_copyin5_c4
    module procedure :: gpufortrt_copyin6_c4
    module procedure :: gpufortrt_copyin7_c4
    module procedure :: gpufortrt_copyin0_c8
    module procedure :: gpufortrt_copyin1_c8
    module procedure :: gpufortrt_copyin2_c8
    module procedure :: gpufortrt_copyin3_c8
    module procedure :: gpufortrt_copyin4_c8
    module procedure :: gpufortrt_copyin5_c8
    module procedure :: gpufortrt_copyin6_c8
    module procedure :: gpufortrt_copyin7_c8
  end interface



  interface gpufortrt_delete
    module procedure :: gpufortrt_delete_b
    module procedure :: gpufortrt_delete0_l1
    module procedure :: gpufortrt_delete1_l1
    module procedure :: gpufortrt_delete2_l1
    module procedure :: gpufortrt_delete3_l1
    module procedure :: gpufortrt_delete4_l1
    module procedure :: gpufortrt_delete5_l1
    module procedure :: gpufortrt_delete6_l1
    module procedure :: gpufortrt_delete7_l1
    module procedure :: gpufortrt_delete0_l4
    module procedure :: gpufortrt_delete1_l4
    module procedure :: gpufortrt_delete2_l4
    module procedure :: gpufortrt_delete3_l4
    module procedure :: gpufortrt_delete4_l4
    module procedure :: gpufortrt_delete5_l4
    module procedure :: gpufortrt_delete6_l4
    module procedure :: gpufortrt_delete7_l4
    module procedure :: gpufortrt_delete0_ch1
    module procedure :: gpufortrt_delete1_ch1
    module procedure :: gpufortrt_delete2_ch1
    module procedure :: gpufortrt_delete3_ch1
    module procedure :: gpufortrt_delete4_ch1
    module procedure :: gpufortrt_delete5_ch1
    module procedure :: gpufortrt_delete6_ch1
    module procedure :: gpufortrt_delete7_ch1
    module procedure :: gpufortrt_delete0_i1
    module procedure :: gpufortrt_delete1_i1
    module procedure :: gpufortrt_delete2_i1
    module procedure :: gpufortrt_delete3_i1
    module procedure :: gpufortrt_delete4_i1
    module procedure :: gpufortrt_delete5_i1
    module procedure :: gpufortrt_delete6_i1
    module procedure :: gpufortrt_delete7_i1
    module procedure :: gpufortrt_delete0_i2
    module procedure :: gpufortrt_delete1_i2
    module procedure :: gpufortrt_delete2_i2
    module procedure :: gpufortrt_delete3_i2
    module procedure :: gpufortrt_delete4_i2
    module procedure :: gpufortrt_delete5_i2
    module procedure :: gpufortrt_delete6_i2
    module procedure :: gpufortrt_delete7_i2
    module procedure :: gpufortrt_delete0_i4
    module procedure :: gpufortrt_delete1_i4
    module procedure :: gpufortrt_delete2_i4
    module procedure :: gpufortrt_delete3_i4
    module procedure :: gpufortrt_delete4_i4
    module procedure :: gpufortrt_delete5_i4
    module procedure :: gpufortrt_delete6_i4
    module procedure :: gpufortrt_delete7_i4
    module procedure :: gpufortrt_delete0_i8
    module procedure :: gpufortrt_delete1_i8
    module procedure :: gpufortrt_delete2_i8
    module procedure :: gpufortrt_delete3_i8
    module procedure :: gpufortrt_delete4_i8
    module procedure :: gpufortrt_delete5_i8
    module procedure :: gpufortrt_delete6_i8
    module procedure :: gpufortrt_delete7_i8
    module procedure :: gpufortrt_delete0_r4
    module procedure :: gpufortrt_delete1_r4
    module procedure :: gpufortrt_delete2_r4
    module procedure :: gpufortrt_delete3_r4
    module procedure :: gpufortrt_delete4_r4
    module procedure :: gpufortrt_delete5_r4
    module procedure :: gpufortrt_delete6_r4
    module procedure :: gpufortrt_delete7_r4
    module procedure :: gpufortrt_delete0_r8
    module procedure :: gpufortrt_delete1_r8
    module procedure :: gpufortrt_delete2_r8
    module procedure :: gpufortrt_delete3_r8
    module procedure :: gpufortrt_delete4_r8
    module procedure :: gpufortrt_delete5_r8
    module procedure :: gpufortrt_delete6_r8
    module procedure :: gpufortrt_delete7_r8
    module procedure :: gpufortrt_delete0_c4
    module procedure :: gpufortrt_delete1_c4
    module procedure :: gpufortrt_delete2_c4
    module procedure :: gpufortrt_delete3_c4
    module procedure :: gpufortrt_delete4_c4
    module procedure :: gpufortrt_delete5_c4
    module procedure :: gpufortrt_delete6_c4
    module procedure :: gpufortrt_delete7_c4
    module procedure :: gpufortrt_delete0_c8
    module procedure :: gpufortrt_delete1_c8
    module procedure :: gpufortrt_delete2_c8
    module procedure :: gpufortrt_delete3_c8
    module procedure :: gpufortrt_delete4_c8
    module procedure :: gpufortrt_delete5_c8
    module procedure :: gpufortrt_delete6_c8
    module procedure :: gpufortrt_delete7_c8
  end interface

  interface gpufortrt_copyout
    module procedure :: gpufortrt_copyout_b
    module procedure :: gpufortrt_copyout0_l1
    module procedure :: gpufortrt_copyout1_l1
    module procedure :: gpufortrt_copyout2_l1
    module procedure :: gpufortrt_copyout3_l1
    module procedure :: gpufortrt_copyout4_l1
    module procedure :: gpufortrt_copyout5_l1
    module procedure :: gpufortrt_copyout6_l1
    module procedure :: gpufortrt_copyout7_l1
    module procedure :: gpufortrt_copyout0_l4
    module procedure :: gpufortrt_copyout1_l4
    module procedure :: gpufortrt_copyout2_l4
    module procedure :: gpufortrt_copyout3_l4
    module procedure :: gpufortrt_copyout4_l4
    module procedure :: gpufortrt_copyout5_l4
    module procedure :: gpufortrt_copyout6_l4
    module procedure :: gpufortrt_copyout7_l4
    module procedure :: gpufortrt_copyout0_ch1
    module procedure :: gpufortrt_copyout1_ch1
    module procedure :: gpufortrt_copyout2_ch1
    module procedure :: gpufortrt_copyout3_ch1
    module procedure :: gpufortrt_copyout4_ch1
    module procedure :: gpufortrt_copyout5_ch1
    module procedure :: gpufortrt_copyout6_ch1
    module procedure :: gpufortrt_copyout7_ch1
    module procedure :: gpufortrt_copyout0_i1
    module procedure :: gpufortrt_copyout1_i1
    module procedure :: gpufortrt_copyout2_i1
    module procedure :: gpufortrt_copyout3_i1
    module procedure :: gpufortrt_copyout4_i1
    module procedure :: gpufortrt_copyout5_i1
    module procedure :: gpufortrt_copyout6_i1
    module procedure :: gpufortrt_copyout7_i1
    module procedure :: gpufortrt_copyout0_i2
    module procedure :: gpufortrt_copyout1_i2
    module procedure :: gpufortrt_copyout2_i2
    module procedure :: gpufortrt_copyout3_i2
    module procedure :: gpufortrt_copyout4_i2
    module procedure :: gpufortrt_copyout5_i2
    module procedure :: gpufortrt_copyout6_i2
    module procedure :: gpufortrt_copyout7_i2
    module procedure :: gpufortrt_copyout0_i4
    module procedure :: gpufortrt_copyout1_i4
    module procedure :: gpufortrt_copyout2_i4
    module procedure :: gpufortrt_copyout3_i4
    module procedure :: gpufortrt_copyout4_i4
    module procedure :: gpufortrt_copyout5_i4
    module procedure :: gpufortrt_copyout6_i4
    module procedure :: gpufortrt_copyout7_i4
    module procedure :: gpufortrt_copyout0_i8
    module procedure :: gpufortrt_copyout1_i8
    module procedure :: gpufortrt_copyout2_i8
    module procedure :: gpufortrt_copyout3_i8
    module procedure :: gpufortrt_copyout4_i8
    module procedure :: gpufortrt_copyout5_i8
    module procedure :: gpufortrt_copyout6_i8
    module procedure :: gpufortrt_copyout7_i8
    module procedure :: gpufortrt_copyout0_r4
    module procedure :: gpufortrt_copyout1_r4
    module procedure :: gpufortrt_copyout2_r4
    module procedure :: gpufortrt_copyout3_r4
    module procedure :: gpufortrt_copyout4_r4
    module procedure :: gpufortrt_copyout5_r4
    module procedure :: gpufortrt_copyout6_r4
    module procedure :: gpufortrt_copyout7_r4
    module procedure :: gpufortrt_copyout0_r8
    module procedure :: gpufortrt_copyout1_r8
    module procedure :: gpufortrt_copyout2_r8
    module procedure :: gpufortrt_copyout3_r8
    module procedure :: gpufortrt_copyout4_r8
    module procedure :: gpufortrt_copyout5_r8
    module procedure :: gpufortrt_copyout6_r8
    module procedure :: gpufortrt_copyout7_r8
    module procedure :: gpufortrt_copyout0_c4
    module procedure :: gpufortrt_copyout1_c4
    module procedure :: gpufortrt_copyout2_c4
    module procedure :: gpufortrt_copyout3_c4
    module procedure :: gpufortrt_copyout4_c4
    module procedure :: gpufortrt_copyout5_c4
    module procedure :: gpufortrt_copyout6_c4
    module procedure :: gpufortrt_copyout7_c4
    module procedure :: gpufortrt_copyout0_c8
    module procedure :: gpufortrt_copyout1_c8
    module procedure :: gpufortrt_copyout2_c8
    module procedure :: gpufortrt_copyout3_c8
    module procedure :: gpufortrt_copyout4_c8
    module procedure :: gpufortrt_copyout5_c8
    module procedure :: gpufortrt_copyout6_c8
    module procedure :: gpufortrt_copyout7_c8
  end interface


  interface gpufortrt_update_self
    module procedure :: gpufortrt_update_self_b
    module procedure :: gpufortrt_update_self0_l1
    module procedure :: gpufortrt_update_self1_l1
    module procedure :: gpufortrt_update_self2_l1
    module procedure :: gpufortrt_update_self3_l1
    module procedure :: gpufortrt_update_self4_l1
    module procedure :: gpufortrt_update_self5_l1
    module procedure :: gpufortrt_update_self6_l1
    module procedure :: gpufortrt_update_self7_l1
    module procedure :: gpufortrt_update_self0_l4
    module procedure :: gpufortrt_update_self1_l4
    module procedure :: gpufortrt_update_self2_l4
    module procedure :: gpufortrt_update_self3_l4
    module procedure :: gpufortrt_update_self4_l4
    module procedure :: gpufortrt_update_self5_l4
    module procedure :: gpufortrt_update_self6_l4
    module procedure :: gpufortrt_update_self7_l4
    module procedure :: gpufortrt_update_self0_ch1
    module procedure :: gpufortrt_update_self1_ch1
    module procedure :: gpufortrt_update_self2_ch1
    module procedure :: gpufortrt_update_self3_ch1
    module procedure :: gpufortrt_update_self4_ch1
    module procedure :: gpufortrt_update_self5_ch1
    module procedure :: gpufortrt_update_self6_ch1
    module procedure :: gpufortrt_update_self7_ch1
    module procedure :: gpufortrt_update_self0_i1
    module procedure :: gpufortrt_update_self1_i1
    module procedure :: gpufortrt_update_self2_i1
    module procedure :: gpufortrt_update_self3_i1
    module procedure :: gpufortrt_update_self4_i1
    module procedure :: gpufortrt_update_self5_i1
    module procedure :: gpufortrt_update_self6_i1
    module procedure :: gpufortrt_update_self7_i1
    module procedure :: gpufortrt_update_self0_i2
    module procedure :: gpufortrt_update_self1_i2
    module procedure :: gpufortrt_update_self2_i2
    module procedure :: gpufortrt_update_self3_i2
    module procedure :: gpufortrt_update_self4_i2
    module procedure :: gpufortrt_update_self5_i2
    module procedure :: gpufortrt_update_self6_i2
    module procedure :: gpufortrt_update_self7_i2
    module procedure :: gpufortrt_update_self0_i4
    module procedure :: gpufortrt_update_self1_i4
    module procedure :: gpufortrt_update_self2_i4
    module procedure :: gpufortrt_update_self3_i4
    module procedure :: gpufortrt_update_self4_i4
    module procedure :: gpufortrt_update_self5_i4
    module procedure :: gpufortrt_update_self6_i4
    module procedure :: gpufortrt_update_self7_i4
    module procedure :: gpufortrt_update_self0_i8
    module procedure :: gpufortrt_update_self1_i8
    module procedure :: gpufortrt_update_self2_i8
    module procedure :: gpufortrt_update_self3_i8
    module procedure :: gpufortrt_update_self4_i8
    module procedure :: gpufortrt_update_self5_i8
    module procedure :: gpufortrt_update_self6_i8
    module procedure :: gpufortrt_update_self7_i8
    module procedure :: gpufortrt_update_self0_r4
    module procedure :: gpufortrt_update_self1_r4
    module procedure :: gpufortrt_update_self2_r4
    module procedure :: gpufortrt_update_self3_r4
    module procedure :: gpufortrt_update_self4_r4
    module procedure :: gpufortrt_update_self5_r4
    module procedure :: gpufortrt_update_self6_r4
    module procedure :: gpufortrt_update_self7_r4
    module procedure :: gpufortrt_update_self0_r8
    module procedure :: gpufortrt_update_self1_r8
    module procedure :: gpufortrt_update_self2_r8
    module procedure :: gpufortrt_update_self3_r8
    module procedure :: gpufortrt_update_self4_r8
    module procedure :: gpufortrt_update_self5_r8
    module procedure :: gpufortrt_update_self6_r8
    module procedure :: gpufortrt_update_self7_r8
    module procedure :: gpufortrt_update_self0_c4
    module procedure :: gpufortrt_update_self1_c4
    module procedure :: gpufortrt_update_self2_c4
    module procedure :: gpufortrt_update_self3_c4
    module procedure :: gpufortrt_update_self4_c4
    module procedure :: gpufortrt_update_self5_c4
    module procedure :: gpufortrt_update_self6_c4
    module procedure :: gpufortrt_update_self7_c4
    module procedure :: gpufortrt_update_self0_c8
    module procedure :: gpufortrt_update_self1_c8
    module procedure :: gpufortrt_update_self2_c8
    module procedure :: gpufortrt_update_self3_c8
    module procedure :: gpufortrt_update_self4_c8
    module procedure :: gpufortrt_update_self5_c8
    module procedure :: gpufortrt_update_self6_c8
    module procedure :: gpufortrt_update_self7_c8
  end interface

  interface gpufortrt_update_device
    module procedure :: gpufortrt_update_device_b
    module procedure :: gpufortrt_update_device0_l1
    module procedure :: gpufortrt_update_device1_l1
    module procedure :: gpufortrt_update_device2_l1
    module procedure :: gpufortrt_update_device3_l1
    module procedure :: gpufortrt_update_device4_l1
    module procedure :: gpufortrt_update_device5_l1
    module procedure :: gpufortrt_update_device6_l1
    module procedure :: gpufortrt_update_device7_l1
    module procedure :: gpufortrt_update_device0_l4
    module procedure :: gpufortrt_update_device1_l4
    module procedure :: gpufortrt_update_device2_l4
    module procedure :: gpufortrt_update_device3_l4
    module procedure :: gpufortrt_update_device4_l4
    module procedure :: gpufortrt_update_device5_l4
    module procedure :: gpufortrt_update_device6_l4
    module procedure :: gpufortrt_update_device7_l4
    module procedure :: gpufortrt_update_device0_ch1
    module procedure :: gpufortrt_update_device1_ch1
    module procedure :: gpufortrt_update_device2_ch1
    module procedure :: gpufortrt_update_device3_ch1
    module procedure :: gpufortrt_update_device4_ch1
    module procedure :: gpufortrt_update_device5_ch1
    module procedure :: gpufortrt_update_device6_ch1
    module procedure :: gpufortrt_update_device7_ch1
    module procedure :: gpufortrt_update_device0_i1
    module procedure :: gpufortrt_update_device1_i1
    module procedure :: gpufortrt_update_device2_i1
    module procedure :: gpufortrt_update_device3_i1
    module procedure :: gpufortrt_update_device4_i1
    module procedure :: gpufortrt_update_device5_i1
    module procedure :: gpufortrt_update_device6_i1
    module procedure :: gpufortrt_update_device7_i1
    module procedure :: gpufortrt_update_device0_i2
    module procedure :: gpufortrt_update_device1_i2
    module procedure :: gpufortrt_update_device2_i2
    module procedure :: gpufortrt_update_device3_i2
    module procedure :: gpufortrt_update_device4_i2
    module procedure :: gpufortrt_update_device5_i2
    module procedure :: gpufortrt_update_device6_i2
    module procedure :: gpufortrt_update_device7_i2
    module procedure :: gpufortrt_update_device0_i4
    module procedure :: gpufortrt_update_device1_i4
    module procedure :: gpufortrt_update_device2_i4
    module procedure :: gpufortrt_update_device3_i4
    module procedure :: gpufortrt_update_device4_i4
    module procedure :: gpufortrt_update_device5_i4
    module procedure :: gpufortrt_update_device6_i4
    module procedure :: gpufortrt_update_device7_i4
    module procedure :: gpufortrt_update_device0_i8
    module procedure :: gpufortrt_update_device1_i8
    module procedure :: gpufortrt_update_device2_i8
    module procedure :: gpufortrt_update_device3_i8
    module procedure :: gpufortrt_update_device4_i8
    module procedure :: gpufortrt_update_device5_i8
    module procedure :: gpufortrt_update_device6_i8
    module procedure :: gpufortrt_update_device7_i8
    module procedure :: gpufortrt_update_device0_r4
    module procedure :: gpufortrt_update_device1_r4
    module procedure :: gpufortrt_update_device2_r4
    module procedure :: gpufortrt_update_device3_r4
    module procedure :: gpufortrt_update_device4_r4
    module procedure :: gpufortrt_update_device5_r4
    module procedure :: gpufortrt_update_device6_r4
    module procedure :: gpufortrt_update_device7_r4
    module procedure :: gpufortrt_update_device0_r8
    module procedure :: gpufortrt_update_device1_r8
    module procedure :: gpufortrt_update_device2_r8
    module procedure :: gpufortrt_update_device3_r8
    module procedure :: gpufortrt_update_device4_r8
    module procedure :: gpufortrt_update_device5_r8
    module procedure :: gpufortrt_update_device6_r8
    module procedure :: gpufortrt_update_device7_r8
    module procedure :: gpufortrt_update_device0_c4
    module procedure :: gpufortrt_update_device1_c4
    module procedure :: gpufortrt_update_device2_c4
    module procedure :: gpufortrt_update_device3_c4
    module procedure :: gpufortrt_update_device4_c4
    module procedure :: gpufortrt_update_device5_c4
    module procedure :: gpufortrt_update_device6_c4
    module procedure :: gpufortrt_update_device7_c4
    module procedure :: gpufortrt_update_device0_c8
    module procedure :: gpufortrt_update_device1_c8
    module procedure :: gpufortrt_update_device2_c8
    module procedure :: gpufortrt_update_device3_c8
    module procedure :: gpufortrt_update_device4_c8
    module procedure :: gpufortrt_update_device5_c8
    module procedure :: gpufortrt_update_device6_c8
    module procedure :: gpufortrt_update_device7_c8
  end interface


  interface gpufortrt_use_device
    module procedure :: gpufortrt_use_device0_l1
    module procedure :: gpufortrt_use_device0_l4
    module procedure :: gpufortrt_use_device0_ch1
    module procedure :: gpufortrt_use_device0_i1
    module procedure :: gpufortrt_use_device0_i2
    module procedure :: gpufortrt_use_device0_i4
    module procedure :: gpufortrt_use_device0_i8
    module procedure :: gpufortrt_use_device0_r4
    module procedure :: gpufortrt_use_device0_r8
    module procedure :: gpufortrt_use_device0_c4
    module procedure :: gpufortrt_use_device0_c8
    module procedure :: gpufortrt_use_device1_l1
    module procedure :: gpufortrt_use_device1_l4
    module procedure :: gpufortrt_use_device1_ch1
    module procedure :: gpufortrt_use_device1_i1
    module procedure :: gpufortrt_use_device1_i2
    module procedure :: gpufortrt_use_device1_i4
    module procedure :: gpufortrt_use_device1_i8
    module procedure :: gpufortrt_use_device1_r4
    module procedure :: gpufortrt_use_device1_r8
    module procedure :: gpufortrt_use_device1_c4
    module procedure :: gpufortrt_use_device1_c8
    module procedure :: gpufortrt_use_device2_l1
    module procedure :: gpufortrt_use_device2_l4
    module procedure :: gpufortrt_use_device2_ch1
    module procedure :: gpufortrt_use_device2_i1
    module procedure :: gpufortrt_use_device2_i2
    module procedure :: gpufortrt_use_device2_i4
    module procedure :: gpufortrt_use_device2_i8
    module procedure :: gpufortrt_use_device2_r4
    module procedure :: gpufortrt_use_device2_r8
    module procedure :: gpufortrt_use_device2_c4
    module procedure :: gpufortrt_use_device2_c8
    module procedure :: gpufortrt_use_device3_l1
    module procedure :: gpufortrt_use_device3_l4
    module procedure :: gpufortrt_use_device3_ch1
    module procedure :: gpufortrt_use_device3_i1
    module procedure :: gpufortrt_use_device3_i2
    module procedure :: gpufortrt_use_device3_i4
    module procedure :: gpufortrt_use_device3_i8
    module procedure :: gpufortrt_use_device3_r4
    module procedure :: gpufortrt_use_device3_r8
    module procedure :: gpufortrt_use_device3_c4
    module procedure :: gpufortrt_use_device3_c8
    module procedure :: gpufortrt_use_device4_l1
    module procedure :: gpufortrt_use_device4_l4
    module procedure :: gpufortrt_use_device4_ch1
    module procedure :: gpufortrt_use_device4_i1
    module procedure :: gpufortrt_use_device4_i2
    module procedure :: gpufortrt_use_device4_i4
    module procedure :: gpufortrt_use_device4_i8
    module procedure :: gpufortrt_use_device4_r4
    module procedure :: gpufortrt_use_device4_r8
    module procedure :: gpufortrt_use_device4_c4
    module procedure :: gpufortrt_use_device4_c8
    module procedure :: gpufortrt_use_device5_l1
    module procedure :: gpufortrt_use_device5_l4
    module procedure :: gpufortrt_use_device5_ch1
    module procedure :: gpufortrt_use_device5_i1
    module procedure :: gpufortrt_use_device5_i2
    module procedure :: gpufortrt_use_device5_i4
    module procedure :: gpufortrt_use_device5_i8
    module procedure :: gpufortrt_use_device5_r4
    module procedure :: gpufortrt_use_device5_r8
    module procedure :: gpufortrt_use_device5_c4
    module procedure :: gpufortrt_use_device5_c8
    module procedure :: gpufortrt_use_device6_l1
    module procedure :: gpufortrt_use_device6_l4
    module procedure :: gpufortrt_use_device6_ch1
    module procedure :: gpufortrt_use_device6_i1
    module procedure :: gpufortrt_use_device6_i2
    module procedure :: gpufortrt_use_device6_i4
    module procedure :: gpufortrt_use_device6_i8
    module procedure :: gpufortrt_use_device6_r4
    module procedure :: gpufortrt_use_device6_r8
    module procedure :: gpufortrt_use_device6_c4
    module procedure :: gpufortrt_use_device6_c8
    module procedure :: gpufortrt_use_device7_l1
    module procedure :: gpufortrt_use_device7_l4
    module procedure :: gpufortrt_use_device7_ch1
    module procedure :: gpufortrt_use_device7_i1
    module procedure :: gpufortrt_use_device7_i2
    module procedure :: gpufortrt_use_device7_i4
    module procedure :: gpufortrt_use_device7_i8
    module procedure :: gpufortrt_use_device7_r4
    module procedure :: gpufortrt_use_device7_r8
    module procedure :: gpufortrt_use_device7_c4
    module procedure :: gpufortrt_use_device7_c8
  end interface


  interface gpufortrt_deviceptr
    module procedure :: gpufortrt_deviceptr0_l1
    module procedure :: gpufortrt_deviceptr1_l1
    module procedure :: gpufortrt_deviceptr2_l1
    module procedure :: gpufortrt_deviceptr3_l1
    module procedure :: gpufortrt_deviceptr4_l1
    module procedure :: gpufortrt_deviceptr5_l1
    module procedure :: gpufortrt_deviceptr6_l1
    module procedure :: gpufortrt_deviceptr7_l1
    module procedure :: gpufortrt_deviceptr0_l4
    module procedure :: gpufortrt_deviceptr1_l4
    module procedure :: gpufortrt_deviceptr2_l4
    module procedure :: gpufortrt_deviceptr3_l4
    module procedure :: gpufortrt_deviceptr4_l4
    module procedure :: gpufortrt_deviceptr5_l4
    module procedure :: gpufortrt_deviceptr6_l4
    module procedure :: gpufortrt_deviceptr7_l4
    module procedure :: gpufortrt_deviceptr0_ch1
    module procedure :: gpufortrt_deviceptr1_ch1
    module procedure :: gpufortrt_deviceptr2_ch1
    module procedure :: gpufortrt_deviceptr3_ch1
    module procedure :: gpufortrt_deviceptr4_ch1
    module procedure :: gpufortrt_deviceptr5_ch1
    module procedure :: gpufortrt_deviceptr6_ch1
    module procedure :: gpufortrt_deviceptr7_ch1
    module procedure :: gpufortrt_deviceptr0_i1
    module procedure :: gpufortrt_deviceptr1_i1
    module procedure :: gpufortrt_deviceptr2_i1
    module procedure :: gpufortrt_deviceptr3_i1
    module procedure :: gpufortrt_deviceptr4_i1
    module procedure :: gpufortrt_deviceptr5_i1
    module procedure :: gpufortrt_deviceptr6_i1
    module procedure :: gpufortrt_deviceptr7_i1
    module procedure :: gpufortrt_deviceptr0_i2
    module procedure :: gpufortrt_deviceptr1_i2
    module procedure :: gpufortrt_deviceptr2_i2
    module procedure :: gpufortrt_deviceptr3_i2
    module procedure :: gpufortrt_deviceptr4_i2
    module procedure :: gpufortrt_deviceptr5_i2
    module procedure :: gpufortrt_deviceptr6_i2
    module procedure :: gpufortrt_deviceptr7_i2
    module procedure :: gpufortrt_deviceptr0_i4
    module procedure :: gpufortrt_deviceptr1_i4
    module procedure :: gpufortrt_deviceptr2_i4
    module procedure :: gpufortrt_deviceptr3_i4
    module procedure :: gpufortrt_deviceptr4_i4
    module procedure :: gpufortrt_deviceptr5_i4
    module procedure :: gpufortrt_deviceptr6_i4
    module procedure :: gpufortrt_deviceptr7_i4
    module procedure :: gpufortrt_deviceptr0_i8
    module procedure :: gpufortrt_deviceptr1_i8
    module procedure :: gpufortrt_deviceptr2_i8
    module procedure :: gpufortrt_deviceptr3_i8
    module procedure :: gpufortrt_deviceptr4_i8
    module procedure :: gpufortrt_deviceptr5_i8
    module procedure :: gpufortrt_deviceptr6_i8
    module procedure :: gpufortrt_deviceptr7_i8
    module procedure :: gpufortrt_deviceptr0_r4
    module procedure :: gpufortrt_deviceptr1_r4
    module procedure :: gpufortrt_deviceptr2_r4
    module procedure :: gpufortrt_deviceptr3_r4
    module procedure :: gpufortrt_deviceptr4_r4
    module procedure :: gpufortrt_deviceptr5_r4
    module procedure :: gpufortrt_deviceptr6_r4
    module procedure :: gpufortrt_deviceptr7_r4
    module procedure :: gpufortrt_deviceptr0_r8
    module procedure :: gpufortrt_deviceptr1_r8
    module procedure :: gpufortrt_deviceptr2_r8
    module procedure :: gpufortrt_deviceptr3_r8
    module procedure :: gpufortrt_deviceptr4_r8
    module procedure :: gpufortrt_deviceptr5_r8
    module procedure :: gpufortrt_deviceptr6_r8
    module procedure :: gpufortrt_deviceptr7_r8
    module procedure :: gpufortrt_deviceptr0_c4
    module procedure :: gpufortrt_deviceptr1_c4
    module procedure :: gpufortrt_deviceptr2_c4
    module procedure :: gpufortrt_deviceptr3_c4
    module procedure :: gpufortrt_deviceptr4_c4
    module procedure :: gpufortrt_deviceptr5_c4
    module procedure :: gpufortrt_deviceptr6_c4
    module procedure :: gpufortrt_deviceptr7_c4
    module procedure :: gpufortrt_deviceptr0_c8
    module procedure :: gpufortrt_deviceptr1_c8
    module procedure :: gpufortrt_deviceptr2_c8
    module procedure :: gpufortrt_deviceptr3_c8
    module procedure :: gpufortrt_deviceptr4_c8
    module procedure :: gpufortrt_deviceptr5_c8
    module procedure :: gpufortrt_deviceptr6_c8
    module procedure :: gpufortrt_deviceptr7_c8
  end interface


contains

  function gpufortrt_map_present_b(hostptr,num_bytes,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    type(c_ptr),intent(in) :: hostptr
    integer(c_size_t),intent(in),optional :: num_bytes 
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    call gpufortrt_mapping_init(retval,hostptr,num_bytes,&
           gpufortrt_map_kind_present,never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present0_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present1_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present2_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present3_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present4_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present5_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present6_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present7_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present0_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present1_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present2_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present3_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present4_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present5_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present6_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present7_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present0_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present1_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present2_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present3_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present4_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present5_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present6_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present7_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present0_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present1_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present2_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present3_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present4_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present5_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present6_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present7_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present0_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present1_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present2_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present3_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present4_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present5_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present6_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present7_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present0_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present1_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present2_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present3_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present4_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present5_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present6_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present7_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present0_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(8,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present1_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present2_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present3_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present4_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present5_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present6_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present7_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present0_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present1_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present2_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present3_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present4_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present5_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present6_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present7_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present0_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(8,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present1_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present2_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present3_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present4_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present5_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present6_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present7_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present0_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2*4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present1_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present2_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present3_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present4_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present5_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present6_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present7_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present0_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2*8,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present1_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present2_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present3_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present4_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present5_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present6_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_present7_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_present_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

    function gpufortrt_map_no_create_b(hostptr,num_bytes,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    type(c_ptr),intent(in) :: hostptr
    integer(c_size_t),intent(in),optional :: num_bytes 
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    call gpufortrt_mapping_init(retval,hostptr,num_bytes,&
           gpufortrt_map_kind_no_create,never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create0_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create1_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create2_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create3_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create4_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create5_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create6_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create7_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create0_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create1_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create2_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create3_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create4_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create5_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create6_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create7_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create0_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create1_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create2_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create3_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create4_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create5_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create6_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create7_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create0_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create1_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create2_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create3_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create4_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create5_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create6_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create7_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create0_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create1_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create2_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create3_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create4_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create5_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create6_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create7_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create0_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create1_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create2_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create3_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create4_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create5_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create6_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create7_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create0_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(8,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create1_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create2_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create3_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create4_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create5_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create6_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create7_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create0_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create1_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create2_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create3_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create4_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create5_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create6_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create7_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create0_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(8,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create1_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create2_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create3_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create4_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create5_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create6_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create7_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create0_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2*4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create1_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create2_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create3_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create4_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create5_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create6_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create7_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create0_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2*8,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create1_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create2_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create3_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create4_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create5_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create6_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_no_create7_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_no_create_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

    function gpufortrt_map_create_b(hostptr,num_bytes,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    type(c_ptr),intent(in) :: hostptr
    integer(c_size_t),intent(in),optional :: num_bytes 
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    call gpufortrt_mapping_init(retval,hostptr,num_bytes,&
           gpufortrt_map_kind_create,never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create0_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create1_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create2_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create3_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create4_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create5_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create6_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create7_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create0_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create1_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create2_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create3_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create4_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create5_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create6_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create7_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create0_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create1_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create2_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create3_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create4_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create5_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create6_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create7_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create0_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create1_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create2_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create3_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create4_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create5_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create6_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create7_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create0_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create1_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create2_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create3_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create4_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create5_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create6_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create7_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create0_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create1_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create2_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create3_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create4_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create5_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create6_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create7_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create0_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(8,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create1_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create2_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create3_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create4_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create5_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create6_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create7_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create0_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create1_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create2_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create3_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create4_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create5_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create6_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create7_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create0_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(8,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create1_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create2_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create3_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create4_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create5_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create6_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create7_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create0_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2*4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create1_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create2_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create3_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create4_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create5_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create6_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create7_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create0_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2*8,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create1_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create2_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create3_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create4_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create5_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create6_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_create7_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_create_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

    function gpufortrt_map_copyin_b(hostptr,num_bytes,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    type(c_ptr),intent(in) :: hostptr
    integer(c_size_t),intent(in),optional :: num_bytes 
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    call gpufortrt_mapping_init(retval,hostptr,num_bytes,&
           gpufortrt_map_kind_copyin,never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin0_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin1_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin2_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin3_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin4_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin5_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin6_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin7_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin0_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin1_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin2_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin3_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin4_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin5_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin6_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin7_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin0_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin1_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin2_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin3_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin4_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin5_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin6_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin7_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin0_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin1_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin2_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin3_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin4_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin5_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin6_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin7_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin0_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin1_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin2_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin3_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin4_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin5_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin6_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin7_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin0_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin1_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin2_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin3_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin4_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin5_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin6_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin7_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin0_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(8,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin1_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin2_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin3_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin4_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin5_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin6_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin7_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin0_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin1_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin2_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin3_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin4_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin5_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin6_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin7_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin0_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(8,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin1_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin2_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin3_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin4_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin5_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin6_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin7_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin0_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2*4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin1_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin2_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin3_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin4_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin5_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin6_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin7_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin0_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2*8,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin1_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin2_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin3_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin4_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin5_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin6_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyin7_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyin_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

    function gpufortrt_map_copy_b(hostptr,num_bytes,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    type(c_ptr),intent(in) :: hostptr
    integer(c_size_t),intent(in),optional :: num_bytes 
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    call gpufortrt_mapping_init(retval,hostptr,num_bytes,&
           gpufortrt_map_kind_copy,never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy0_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy1_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy2_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy3_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy4_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy5_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy6_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy7_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy0_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy1_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy2_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy3_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy4_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy5_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy6_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy7_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy0_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy1_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy2_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy3_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy4_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy5_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy6_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy7_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy0_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy1_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy2_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy3_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy4_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy5_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy6_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy7_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy0_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy1_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy2_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy3_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy4_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy5_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy6_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy7_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy0_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy1_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy2_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy3_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy4_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy5_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy6_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy7_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy0_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(8,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy1_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy2_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy3_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy4_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy5_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy6_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy7_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy0_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy1_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy2_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy3_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy4_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy5_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy6_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy7_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy0_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(8,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy1_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy2_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy3_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy4_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy5_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy6_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy7_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy0_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2*4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy1_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy2_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy3_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy4_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy5_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy6_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy7_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy0_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2*8,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy1_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy2_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy3_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy4_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy5_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy6_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copy7_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copy_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

    function gpufortrt_map_copyout_b(hostptr,num_bytes,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    type(c_ptr),intent(in) :: hostptr
    integer(c_size_t),intent(in),optional :: num_bytes 
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    call gpufortrt_mapping_init(retval,hostptr,num_bytes,&
           gpufortrt_map_kind_copyout,never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout0_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout1_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout2_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout3_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout4_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout5_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout6_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout7_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout0_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout1_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout2_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout3_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout4_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout5_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout6_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout7_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout0_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout1_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout2_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout3_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout4_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout5_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout6_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout7_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout0_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout1_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout2_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout3_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout4_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout5_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout6_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout7_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout0_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout1_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout2_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout3_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout4_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout5_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout6_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout7_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout0_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout1_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout2_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout3_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout4_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout5_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout6_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout7_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout0_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(8,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout1_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout2_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout3_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout4_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout5_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout6_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout7_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout0_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout1_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout2_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout3_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout4_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout5_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout6_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout7_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout0_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(8,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout1_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout2_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout3_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout4_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout5_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout6_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout7_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout0_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2*4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout1_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout2_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout3_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout4_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout5_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout6_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout7_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout0_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2*8,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout1_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout2_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout3_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout4_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout5_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout6_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_copyout7_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_copyout_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

    function gpufortrt_map_delete_b(hostptr,num_bytes,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    type(c_ptr),intent(in) :: hostptr
    integer(c_size_t),intent(in),optional :: num_bytes 
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    call gpufortrt_mapping_init(retval,hostptr,num_bytes,&
           gpufortrt_map_kind_delete,never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete0_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete1_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete2_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete3_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete4_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete5_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete6_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete7_l1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete0_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete1_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete2_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete3_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete4_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete5_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete6_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete7_l4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete0_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete1_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete2_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete3_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete4_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete5_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete6_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete7_ch1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete0_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete1_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete2_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete3_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete4_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete5_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete6_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete7_i1(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete0_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete1_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete2_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete3_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete4_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete5_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete6_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete7_i2(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete0_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete1_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete2_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete3_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete4_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete5_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete6_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete7_i4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete0_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(8,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete1_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete2_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete3_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete4_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete5_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete6_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete7_i8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete0_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete1_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete2_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete3_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete4_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete5_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete6_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete7_r4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete0_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(8,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete1_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete2_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete3_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete4_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete5_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete6_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete7_r8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete0_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2*4,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete1_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete2_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete3_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete4_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete5_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete6_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete7_c4(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),never_deallocate)
  end function

   !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete0_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2*8,c_size_t),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete1_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete2_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete3_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete4_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete5_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete6_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

  !> \note never_deallocate only has effect on create,copyin,copyout, and copy mappings.
  function gpufortrt_map_delete7_c8(hostptr,never_deallocate) result(retval)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    !
    type(gpufortrt_mapping_t) :: retval
    !
    retval = gpufortrt_map_delete_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),never_deallocate)
  end function

     

  !> Map and directly return the corresponding deviceptr.
  function gpufortrt_create_b(hostptr,num_bytes,never_deallocate,&
      async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    type(c_ptr),intent(in) :: hostptr
    integer(c_size_t),intent(in) :: num_bytes
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    interface
      function gpufortrt_create_c_impl(hostptr,num_bytes,never_deallocate) &
          bind(c,name="gpufortrt_create") result(deviceptr)
        use iso_c_binding
        use gpufortrt_types
        implicit none
        type(c_ptr),value,intent(in) :: hostptr
        integer(c_size_t),value,intent(in) :: num_bytes
        logical(c_bool),value,intent(in) :: never_deallocate
        !
        type(c_ptr) :: deviceptr
      end function
      function gpufortrt_create_async_c_impl(hostptr,num_bytes,never_deallocate,async_arg) &
          bind(c,name="gpufortrt_create_async") result(deviceptr)
        use iso_c_binding
        use gpufortrt_types
        implicit none
        type(c_ptr),value,intent(in) :: hostptr
        integer(c_size_t),value,intent(in) :: num_bytes
        logical(c_bool),value,intent(in) :: never_deallocate
        integer(gpufortrt_handle_kind),value,intent(in) :: async_arg
        !
        type(c_ptr) :: deviceptr
      end function
    end interface
    !
    logical(c_bool) :: opt_never_deallocate
    !
    opt_never_deallocate = .false._c_bool
    if ( present(never_deallocate) ) opt_never_deallocate = never_deallocate
    if ( present(async_arg) ) then
      deviceptr = gpufortrt_create_async_c_impl(hostptr,num_bytes,opt_never_deallocate,async_arg)
    else
      deviceptr = gpufortrt_create_c_impl(hostptr,num_bytes,opt_never_deallocate)
    endif
  end function

  !> Map and directly return the corresponding deviceptr.
  function gpufortrt_copyin_b(hostptr,num_bytes,never_deallocate,&
      async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    type(c_ptr),intent(in) :: hostptr
    integer(c_size_t),intent(in) :: num_bytes
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    interface
      function gpufortrt_copyin_c_impl(hostptr,num_bytes,never_deallocate) &
          bind(c,name="gpufortrt_copyin") result(deviceptr)
        use iso_c_binding
        use gpufortrt_types
        implicit none
        type(c_ptr),value,intent(in) :: hostptr
        integer(c_size_t),value,intent(in) :: num_bytes
        logical(c_bool),value,intent(in) :: never_deallocate
        !
        type(c_ptr) :: deviceptr
      end function
      function gpufortrt_copyin_async_c_impl(hostptr,num_bytes,never_deallocate,async_arg) &
          bind(c,name="gpufortrt_copyin_async") result(deviceptr)
        use iso_c_binding
        use gpufortrt_types
        implicit none
        type(c_ptr),value,intent(in) :: hostptr
        integer(c_size_t),value,intent(in) :: num_bytes
        logical(c_bool),value,intent(in) :: never_deallocate
        integer(gpufortrt_handle_kind),value,intent(in) :: async_arg
        !
        type(c_ptr) :: deviceptr
      end function
    end interface
    !
    logical(c_bool) :: opt_never_deallocate
    !
    opt_never_deallocate = .false._c_bool
    if ( present(never_deallocate) ) opt_never_deallocate = never_deallocate
    if ( present(async_arg) ) then
      deviceptr = gpufortrt_copyin_async_c_impl(hostptr,num_bytes,opt_never_deallocate,async_arg)
    else
      deviceptr = gpufortrt_copyin_c_impl(hostptr,num_bytes,opt_never_deallocate)
    endif
  end function


  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran scalar arguments)
  function gpufortrt_create0_l1(hostptr,never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(1,kind=c_size_t),&
                                       never_deallocate,async_arg)
  end function

  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create1_l1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create2_l1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create3_l1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create4_l1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create5_l1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create6_l1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create7_l1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran scalar arguments)
  function gpufortrt_create0_l4(hostptr,never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(4,kind=c_size_t),&
                                       never_deallocate,async_arg)
  end function

  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create1_l4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create2_l4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create3_l4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create4_l4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create5_l4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create6_l4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create7_l4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran scalar arguments)
  function gpufortrt_create0_ch1(hostptr,never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(1,kind=c_size_t),&
                                       never_deallocate,async_arg)
  end function

  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create1_ch1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create2_ch1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create3_ch1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create4_ch1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create5_ch1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create6_ch1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create7_ch1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran scalar arguments)
  function gpufortrt_create0_i1(hostptr,never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(1,kind=c_size_t),&
                                       never_deallocate,async_arg)
  end function

  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create1_i1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create2_i1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create3_i1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create4_i1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create5_i1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create6_i1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create7_i1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran scalar arguments)
  function gpufortrt_create0_i2(hostptr,never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(2,kind=c_size_t),&
                                       never_deallocate,async_arg)
  end function

  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create1_i2(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create2_i2(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create3_i2(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create4_i2(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create5_i2(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create6_i2(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create7_i2(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran scalar arguments)
  function gpufortrt_create0_i4(hostptr,never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(4,kind=c_size_t),&
                                       never_deallocate,async_arg)
  end function

  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create1_i4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create2_i4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create3_i4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create4_i4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create5_i4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create6_i4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create7_i4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran scalar arguments)
  function gpufortrt_create0_i8(hostptr,never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(8,kind=c_size_t),&
                                       never_deallocate,async_arg)
  end function

  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create1_i8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create2_i8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create3_i8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create4_i8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create5_i8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create6_i8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create7_i8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran scalar arguments)
  function gpufortrt_create0_r4(hostptr,never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(4,kind=c_size_t),&
                                       never_deallocate,async_arg)
  end function

  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create1_r4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create2_r4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create3_r4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create4_r4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create5_r4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create6_r4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create7_r4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran scalar arguments)
  function gpufortrt_create0_r8(hostptr,never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(8,kind=c_size_t),&
                                       never_deallocate,async_arg)
  end function

  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create1_r8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create2_r8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create3_r8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create4_r8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create5_r8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create6_r8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create7_r8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran scalar arguments)
  function gpufortrt_create0_c4(hostptr,never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(2*4,kind=c_size_t),&
                                       never_deallocate,async_arg)
  end function

  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create1_c4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create2_c4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create3_c4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create4_c4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create5_c4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create6_c4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create7_c4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran scalar arguments)
  function gpufortrt_create0_c8(hostptr,never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(2*8,kind=c_size_t),&
                                       never_deallocate,async_arg)
  end function

  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create1_c8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create2_c8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create3_c8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create4_c8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create5_c8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create6_c8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_create7_c8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_create_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran scalar arguments)
  function gpufortrt_copyin0_l1(hostptr,never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(1,kind=c_size_t),&
                                       never_deallocate,async_arg)
  end function

  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin1_l1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin2_l1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin3_l1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin4_l1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin5_l1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin6_l1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin7_l1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran scalar arguments)
  function gpufortrt_copyin0_l4(hostptr,never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(4,kind=c_size_t),&
                                       never_deallocate,async_arg)
  end function

  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin1_l4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin2_l4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin3_l4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin4_l4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin5_l4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin6_l4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin7_l4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran scalar arguments)
  function gpufortrt_copyin0_ch1(hostptr,never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(1,kind=c_size_t),&
                                       never_deallocate,async_arg)
  end function

  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin1_ch1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin2_ch1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin3_ch1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin4_ch1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin5_ch1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin6_ch1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin7_ch1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran scalar arguments)
  function gpufortrt_copyin0_i1(hostptr,never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(1,kind=c_size_t),&
                                       never_deallocate,async_arg)
  end function

  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin1_i1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin2_i1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin3_i1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin4_i1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin5_i1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin6_i1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin7_i1(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran scalar arguments)
  function gpufortrt_copyin0_i2(hostptr,never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(2,kind=c_size_t),&
                                       never_deallocate,async_arg)
  end function

  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin1_i2(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin2_i2(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin3_i2(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin4_i2(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin5_i2(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin6_i2(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin7_i2(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran scalar arguments)
  function gpufortrt_copyin0_i4(hostptr,never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(4,kind=c_size_t),&
                                       never_deallocate,async_arg)
  end function

  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin1_i4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin2_i4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin3_i4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin4_i4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin5_i4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin6_i4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin7_i4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran scalar arguments)
  function gpufortrt_copyin0_i8(hostptr,never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(8,kind=c_size_t),&
                                       never_deallocate,async_arg)
  end function

  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin1_i8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin2_i8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin3_i8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin4_i8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin5_i8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin6_i8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin7_i8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran scalar arguments)
  function gpufortrt_copyin0_r4(hostptr,never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(4,kind=c_size_t),&
                                       never_deallocate,async_arg)
  end function

  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin1_r4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin2_r4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin3_r4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin4_r4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin5_r4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin6_r4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin7_r4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran scalar arguments)
  function gpufortrt_copyin0_r8(hostptr,never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(8,kind=c_size_t),&
                                       never_deallocate,async_arg)
  end function

  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin1_r8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin2_r8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin3_r8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin4_r8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin5_r8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin6_r8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin7_r8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran scalar arguments)
  function gpufortrt_copyin0_c4(hostptr,never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(2*4,kind=c_size_t),&
                                       never_deallocate,async_arg)
  end function

  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin1_c4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin2_c4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin3_c4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin4_c4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin5_c4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin6_c4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin7_c4(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran scalar arguments)
  function gpufortrt_copyin0_c8(hostptr,never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(2*8,kind=c_size_t),&
                                       never_deallocate,async_arg)
  end function

  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin1_c8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin2_c8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin3_c8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin4_c8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin5_c8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin6_c8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function
  !> Map and directly return the corresponding deviceptr.
  !> (Specialized version for Fortran array arguments)
  function gpufortrt_copyin7_c8(hostptr,&
      never_deallocate,async_arg) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: never_deallocate
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg 
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_copyin_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                       never_deallocate,async_arg)
  end function

  function gpufortrt_present0_l1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(1,kind=c_size_t))
  end function

  function gpufortrt_present1_l1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present2_l1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present3_l1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present4_l1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present5_l1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present6_l1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present7_l1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present0_l4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(4,kind=c_size_t))
  end function

  function gpufortrt_present1_l4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present2_l4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present3_l4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present4_l4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present5_l4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present6_l4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present7_l4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present0_ch1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(1,kind=c_size_t))
  end function

  function gpufortrt_present1_ch1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present2_ch1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present3_ch1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present4_ch1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present5_ch1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present6_ch1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present7_ch1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present0_i1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(1,kind=c_size_t))
  end function

  function gpufortrt_present1_i1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present2_i1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present3_i1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present4_i1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present5_i1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present6_i1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present7_i1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present0_i2(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(2,kind=c_size_t))
  end function

  function gpufortrt_present1_i2(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present2_i2(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present3_i2(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present4_i2(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present5_i2(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present6_i2(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present7_i2(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present0_i4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(4,kind=c_size_t))
  end function

  function gpufortrt_present1_i4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present2_i4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present3_i4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present4_i4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present5_i4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present6_i4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present7_i4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present0_i8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(8,kind=c_size_t))
  end function

  function gpufortrt_present1_i8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present2_i8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present3_i8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present4_i8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present5_i8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present6_i8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present7_i8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present0_r4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(4,kind=c_size_t))
  end function

  function gpufortrt_present1_r4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present2_r4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present3_r4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present4_r4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present5_r4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present6_r4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present7_r4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present0_r8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(8,kind=c_size_t))
  end function

  function gpufortrt_present1_r8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present2_r8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present3_r8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present4_r8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present5_r8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present6_r8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present7_r8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present0_c4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(2*4,kind=c_size_t))
  end function

  function gpufortrt_present1_c4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present2_c4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present3_c4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present4_c4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present5_c4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present6_c4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present7_c4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present0_c8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(2*8,kind=c_size_t))
  end function

  function gpufortrt_present1_c8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present2_c8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present3_c8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present4_c8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present5_c8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present6_c8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr))
  end function

  function gpufortrt_present7_c8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_present_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr))
  end function


  subroutine gpufortrt_delete_b(hostptr,num_bytes,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types, only: gpufortrt_handle_kind
    implicit none
    type(c_ptr), intent(in) :: hostptr
    integer(c_size_t),intent(in),optional :: num_bytes
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    interface
      subroutine gpufortrt_delete_c_impl(hostptr,num_bytes) &
          bind(c,name="gpufortrt_delete")
        use iso_c_binding
        implicit none
        type(c_ptr),value,intent(in) :: hostptr
        integer(c_size_t),value,intent(in) :: num_bytes
      end subroutine
      subroutine gpufortrt_delete_finalize_c_impl(hostptr,num_bytes) &
          bind(c,name="gpufortrt_delete_finalize")
        use iso_c_binding
        implicit none
        type(c_ptr),value,intent(in) :: hostptr
        integer(c_size_t),value,intent(in) :: num_bytes
      end subroutine
      subroutine gpufortrt_delete_async_c_impl(hostptr,num_bytes,async_arg) &
          bind(c,name="gpufortrt_delete_async")
        use iso_c_binding
        use gpufortrt_types, only: gpufortrt_handle_kind
        implicit none
        type(c_ptr),value,intent(in) :: hostptr
        integer(c_size_t),value,intent(in) :: num_bytes
        integer(gpufortrt_handle_kind),value,intent(in) :: async_arg
      end subroutine
      subroutine gpufortrt_delete_finalize_async_c_impl(hostptr,num_bytes,async_arg) &
          bind(c,name="gpufortrt_delete_finalize_async")
        use iso_c_binding
        use gpufortrt_types, only: gpufortrt_handle_kind
        implicit none
        type(c_ptr),value,intent(in) :: hostptr
        integer(c_size_t),value,intent(in) :: num_bytes
        integer(gpufortrt_handle_kind),value,intent(in) :: async_arg
      end subroutine
    end interface
    !
    if ( present(async_arg) ) then
      if ( present(finalize) ) then
        call gpufortrt_delete_finalize_async_c_impl(hostptr,num_bytes,async_arg)
      else
        call gpufortrt_delete_async_c_impl(hostptr,num_bytes,async_arg)
      endif
    else
      if ( present(finalize) ) then
        call gpufortrt_delete_finalize_c_impl(hostptr,num_bytes)
      else
        call gpufortrt_delete_c_impl(hostptr,num_bytes)
      endif
    endif
  end subroutine

  subroutine gpufortrt_copyout_b(hostptr,num_bytes,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types, only: gpufortrt_handle_kind
    implicit none
    type(c_ptr), intent(in) :: hostptr
    integer(c_size_t),intent(in),optional :: num_bytes
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    interface
      subroutine gpufortrt_copyout_c_impl(hostptr,num_bytes) &
          bind(c,name="gpufortrt_copyout")
        use iso_c_binding
        implicit none
        type(c_ptr),value,intent(in) :: hostptr
        integer(c_size_t),value,intent(in) :: num_bytes
      end subroutine
      subroutine gpufortrt_copyout_finalize_c_impl(hostptr,num_bytes) &
          bind(c,name="gpufortrt_copyout_finalize")
        use iso_c_binding
        implicit none
        type(c_ptr),value,intent(in) :: hostptr
        integer(c_size_t),value,intent(in) :: num_bytes
      end subroutine
      subroutine gpufortrt_copyout_async_c_impl(hostptr,num_bytes,async_arg) &
          bind(c,name="gpufortrt_copyout_async")
        use iso_c_binding
        use gpufortrt_types, only: gpufortrt_handle_kind
        implicit none
        type(c_ptr),value,intent(in) :: hostptr
        integer(c_size_t),value,intent(in) :: num_bytes
        integer(gpufortrt_handle_kind),value,intent(in) :: async_arg
      end subroutine
      subroutine gpufortrt_copyout_finalize_async_c_impl(hostptr,num_bytes,async_arg) &
          bind(c,name="gpufortrt_copyout_finalize_async")
        use iso_c_binding
        use gpufortrt_types, only: gpufortrt_handle_kind
        implicit none
        type(c_ptr),value,intent(in) :: hostptr
        integer(c_size_t),value,intent(in) :: num_bytes
        integer(gpufortrt_handle_kind),value,intent(in) :: async_arg
      end subroutine
    end interface
    !
    if ( present(async_arg) ) then
      if ( present(finalize) ) then
        call gpufortrt_copyout_finalize_async_c_impl(hostptr,num_bytes,async_arg)
      else
        call gpufortrt_copyout_async_c_impl(hostptr,num_bytes,async_arg)
      endif
    else
      if ( present(finalize) ) then
        call gpufortrt_copyout_finalize_c_impl(hostptr,num_bytes)
      else
        call gpufortrt_copyout_c_impl(hostptr,num_bytes)
      endif
    endif
  end subroutine


  !> (Specialized version for Fortran scalar arguments)
  subroutine gpufortrt_delete0_l1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(1,kind=c_size_t),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete1_l1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete2_l1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete3_l1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete4_l1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete5_l1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete6_l1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete7_l1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran scalar arguments)
  subroutine gpufortrt_delete0_l4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(4,kind=c_size_t),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete1_l4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete2_l4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete3_l4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete4_l4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete5_l4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete6_l4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete7_l4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran scalar arguments)
  subroutine gpufortrt_delete0_ch1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(1,kind=c_size_t),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete1_ch1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete2_ch1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete3_ch1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete4_ch1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete5_ch1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete6_ch1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete7_ch1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran scalar arguments)
  subroutine gpufortrt_delete0_i1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(1,kind=c_size_t),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete1_i1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete2_i1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete3_i1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete4_i1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete5_i1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete6_i1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete7_i1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran scalar arguments)
  subroutine gpufortrt_delete0_i2(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(2,kind=c_size_t),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete1_i2(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete2_i2(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete3_i2(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete4_i2(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete5_i2(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete6_i2(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete7_i2(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran scalar arguments)
  subroutine gpufortrt_delete0_i4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(4,kind=c_size_t),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete1_i4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete2_i4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete3_i4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete4_i4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete5_i4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete6_i4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete7_i4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran scalar arguments)
  subroutine gpufortrt_delete0_i8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(8,kind=c_size_t),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete1_i8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete2_i8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete3_i8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete4_i8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete5_i8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete6_i8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete7_i8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran scalar arguments)
  subroutine gpufortrt_delete0_r4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(4,kind=c_size_t),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete1_r4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete2_r4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete3_r4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete4_r4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete5_r4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete6_r4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete7_r4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran scalar arguments)
  subroutine gpufortrt_delete0_r8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(8,kind=c_size_t),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete1_r8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete2_r8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete3_r8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete4_r8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete5_r8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete6_r8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete7_r8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran scalar arguments)
  subroutine gpufortrt_delete0_c4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(2*4,kind=c_size_t),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete1_c4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete2_c4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete3_c4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete4_c4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete5_c4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete6_c4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete7_c4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran scalar arguments)
  subroutine gpufortrt_delete0_c8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(2*8,kind=c_size_t),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete1_c8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete2_c8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete3_c8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete4_c8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete5_c8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete6_c8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_delete7_c8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_delete_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran scalar arguments)
  subroutine gpufortrt_copyout0_l1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(1,kind=c_size_t),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout1_l1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout2_l1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout3_l1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout4_l1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout5_l1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout6_l1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout7_l1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran scalar arguments)
  subroutine gpufortrt_copyout0_l4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(4,kind=c_size_t),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout1_l4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout2_l4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout3_l4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout4_l4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout5_l4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout6_l4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout7_l4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran scalar arguments)
  subroutine gpufortrt_copyout0_ch1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(1,kind=c_size_t),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout1_ch1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout2_ch1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout3_ch1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout4_ch1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout5_ch1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout6_ch1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout7_ch1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran scalar arguments)
  subroutine gpufortrt_copyout0_i1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(1,kind=c_size_t),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout1_i1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout2_i1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout3_i1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout4_i1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout5_i1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout6_i1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout7_i1(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(1,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran scalar arguments)
  subroutine gpufortrt_copyout0_i2(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(2,kind=c_size_t),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout1_i2(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout2_i2(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout3_i2(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout4_i2(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout5_i2(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout6_i2(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout7_i2(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(2,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran scalar arguments)
  subroutine gpufortrt_copyout0_i4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(4,kind=c_size_t),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout1_i4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout2_i4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout3_i4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout4_i4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout5_i4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout6_i4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout7_i4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran scalar arguments)
  subroutine gpufortrt_copyout0_i8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(8,kind=c_size_t),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout1_i8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout2_i8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout3_i8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout4_i8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout5_i8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout6_i8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout7_i8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran scalar arguments)
  subroutine gpufortrt_copyout0_r4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(4,kind=c_size_t),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout1_r4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout2_r4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout3_r4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout4_r4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout5_r4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout6_r4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout7_r4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran scalar arguments)
  subroutine gpufortrt_copyout0_r8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(8,kind=c_size_t),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout1_r8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout2_r8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout3_r8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout4_r8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout5_r8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout6_r8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout7_r8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran scalar arguments)
  subroutine gpufortrt_copyout0_c4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(2*4,kind=c_size_t),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout1_c4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout2_c4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout3_c4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout4_c4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout5_c4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout6_c4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout7_c4(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(2*4,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran scalar arguments)
  subroutine gpufortrt_copyout0_c8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(2*8,kind=c_size_t),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout1_c8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout2_c8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout3_c8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout4_c8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout5_c8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout6_c8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine

  !> (Specialized version for Fortran array arguments)
  subroutine gpufortrt_copyout7_c8(hostptr,async_arg,finalize)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    logical,intent(in),optional :: finalize
    !
    call gpufortrt_copyout_b(c_loc(hostptr),int(2*8,kind=c_size_t)*size(hostptr),&
                                async_arg,finalize)
  end subroutine



subroutine gpufortrt_update_self_b(hostptr,num_bytes,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  type(c_ptr),intent(in) :: hostptr
  integer(c_size_t),intent(in) :: num_bytes
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  interface
    subroutine gpufortrt_update_self_c_impl(hostptr,num_bytes,if_arg,if_present_arg) &
            bind(c,name="gpufortrt_update_self")
      use iso_c_binding
      implicit none
      type(c_ptr),value,intent(in) :: hostptr
      integer(c_size_t),value,intent(in) :: num_bytes
      logical(c_bool),value,intent(in) :: if_arg, if_present_arg
    end subroutine
    subroutine gpufortrt_update_self_async_c_impl(hostptr,num_bytes,if_arg,if_present_arg,async_arg) &
            bind(c,name="gpufortrt_update_self_async")
      use iso_c_binding
      use gpufortrt_types
      implicit none
      type(c_ptr),value,intent(in) :: hostptr
      integer(c_size_t),value,intent(in) :: num_bytes
      logical(c_bool),value,intent(in) :: if_arg, if_present_arg
      integer(gpufortrt_handle_kind),value,intent(in) :: async_arg
    end subroutine
  end interface
  logical :: opt_if_arg, opt_if_present_arg
  !
  opt_if_arg = .true.
  opt_if_present_arg = .false.
  if ( present(if_arg) ) opt_if_arg = if_arg
  if ( present(if_present_arg) ) opt_if_present_arg = if_present_arg
  !
  if ( present(async_arg) ) then
    call gpufortrt_update_self_async_c_impl(hostptr,&
                                                       num_bytes,&
                                                       logical(opt_if_arg,c_bool),&
                                                       logical(opt_if_present_arg,c_bool),&
                                                       async_arg)
  else
    call gpufortrt_update_self_c_impl(hostptr,&
                                                 num_bytes,&
                                                 logical(opt_if_arg,c_bool),&
                                                 logical(opt_if_present_arg,c_bool))
  endif
end subroutine

subroutine gpufortrt_update_self0_l1(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  logical(c_bool),target,intent(in) :: hostptr
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(1,c_size_t),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self1_l1(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  logical(c_bool),target,intent(in) :: hostptr(:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self2_l1(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  logical(c_bool),target,intent(in) :: hostptr(:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self3_l1(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  logical(c_bool),target,intent(in) :: hostptr(:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self4_l1(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  logical(c_bool),target,intent(in) :: hostptr(:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self5_l1(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self6_l1(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self7_l1(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self0_l4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  logical,target,intent(in) :: hostptr
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(4,c_size_t),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self1_l4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  logical,target,intent(in) :: hostptr(:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self2_l4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  logical,target,intent(in) :: hostptr(:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self3_l4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  logical,target,intent(in) :: hostptr(:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self4_l4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  logical,target,intent(in) :: hostptr(:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self5_l4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  logical,target,intent(in) :: hostptr(:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self6_l4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  logical,target,intent(in) :: hostptr(:,:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self7_l4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  logical,target,intent(in) :: hostptr(:,:,:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self0_ch1(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  character(c_char),target,intent(in) :: hostptr
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(1,c_size_t),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self1_ch1(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  character(c_char),target,intent(in) :: hostptr(:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self2_ch1(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  character(c_char),target,intent(in) :: hostptr(:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self3_ch1(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  character(c_char),target,intent(in) :: hostptr(:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self4_ch1(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  character(c_char),target,intent(in) :: hostptr(:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self5_ch1(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  character(c_char),target,intent(in) :: hostptr(:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self6_ch1(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self7_ch1(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self0_i1(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_int8_t),target,intent(in) :: hostptr
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(1,c_size_t),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self1_i1(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_int8_t),target,intent(in) :: hostptr(:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self2_i1(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_int8_t),target,intent(in) :: hostptr(:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self3_i1(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_int8_t),target,intent(in) :: hostptr(:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self4_i1(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self5_i1(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self6_i1(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self7_i1(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self0_i2(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_short),target,intent(in) :: hostptr
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(2,c_size_t),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self1_i2(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_short),target,intent(in) :: hostptr(:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self2_i2(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_short),target,intent(in) :: hostptr(:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self3_i2(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_short),target,intent(in) :: hostptr(:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self4_i2(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_short),target,intent(in) :: hostptr(:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self5_i2(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self6_i2(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self7_i2(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self0_i4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_int),target,intent(in) :: hostptr
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(4,c_size_t),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self1_i4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_int),target,intent(in) :: hostptr(:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self2_i4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_int),target,intent(in) :: hostptr(:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self3_i4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_int),target,intent(in) :: hostptr(:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self4_i4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_int),target,intent(in) :: hostptr(:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self5_i4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self6_i4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self7_i4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self0_i8(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_long),target,intent(in) :: hostptr
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(8,c_size_t),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self1_i8(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_long),target,intent(in) :: hostptr(:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self2_i8(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_long),target,intent(in) :: hostptr(:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self3_i8(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_long),target,intent(in) :: hostptr(:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self4_i8(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_long),target,intent(in) :: hostptr(:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self5_i8(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self6_i8(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self7_i8(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self0_r4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  real(c_float),target,intent(in) :: hostptr
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(4,c_size_t),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self1_r4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  real(c_float),target,intent(in) :: hostptr(:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self2_r4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  real(c_float),target,intent(in) :: hostptr(:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self3_r4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  real(c_float),target,intent(in) :: hostptr(:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self4_r4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  real(c_float),target,intent(in) :: hostptr(:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self5_r4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  real(c_float),target,intent(in) :: hostptr(:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self6_r4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self7_r4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self0_r8(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  real(c_double),target,intent(in) :: hostptr
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(8,c_size_t),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self1_r8(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  real(c_double),target,intent(in) :: hostptr(:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self2_r8(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  real(c_double),target,intent(in) :: hostptr(:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self3_r8(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  real(c_double),target,intent(in) :: hostptr(:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self4_r8(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  real(c_double),target,intent(in) :: hostptr(:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self5_r8(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  real(c_double),target,intent(in) :: hostptr(:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self6_r8(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self7_r8(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self0_c4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  complex(c_float_complex),target,intent(in) :: hostptr
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(2*4,c_size_t),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self1_c4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  complex(c_float_complex),target,intent(in) :: hostptr(:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self2_c4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  complex(c_float_complex),target,intent(in) :: hostptr(:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self3_c4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  complex(c_float_complex),target,intent(in) :: hostptr(:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self4_c4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self5_c4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self6_c4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self7_c4(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self0_c8(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  complex(c_double_complex),target,intent(in) :: hostptr
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(2*8,c_size_t),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self1_c8(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  complex(c_double_complex),target,intent(in) :: hostptr(:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self2_c8(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  complex(c_double_complex),target,intent(in) :: hostptr(:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self3_c8(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  complex(c_double_complex),target,intent(in) :: hostptr(:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self4_c8(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self5_c8(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self6_c8(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine

subroutine gpufortrt_update_self7_c8(hostptr,if_arg,if_present_arg,async_arg)
  use iso_c_binding
  use gpufortrt_types
  implicit none
  complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
  logical,intent(in),optional :: if_arg, if_present_arg
  integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
  !
  call gpufortrt_update_self_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
end subroutine



  subroutine gpufortrt_update_device_b(hostptr,num_bytes,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    type(c_ptr),intent(in) :: hostptr
    integer(c_size_t),intent(in) :: num_bytes
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    interface
      subroutine gpufortrt_update_device_c_impl(hostptr,num_bytes,if_arg,if_present_arg) &
              bind(c,name="gpufortrt_update_device")
        use iso_c_binding
        implicit none
        type(c_ptr),value,intent(in) :: hostptr
        integer(c_size_t),value,intent(in) :: num_bytes
        logical(c_bool),value,intent(in) :: if_arg, if_present_arg
      end subroutine
      subroutine gpufortrt_update_device_async_c_impl(hostptr,num_bytes,if_arg,if_present_arg,async_arg) &
              bind(c,name="gpufortrt_update_device_async")
        use iso_c_binding
        use gpufortrt_types
        implicit none
        type(c_ptr),value,intent(in) :: hostptr
        integer(c_size_t),value,intent(in) :: num_bytes
        logical(c_bool),value,intent(in) :: if_arg, if_present_arg
        integer(gpufortrt_handle_kind),value,intent(in) :: async_arg
      end subroutine
    end interface
    logical :: opt_if_arg, opt_if_present_arg
    !
    opt_if_arg = .true.
    opt_if_present_arg = .false.
    if ( present(if_arg) ) opt_if_arg = if_arg
    if ( present(if_present_arg) ) opt_if_present_arg = if_present_arg
    !
    if ( present(async_arg) ) then
      call gpufortrt_update_device_async_c_impl(hostptr,&
                                                         num_bytes,&
                                                         logical(opt_if_arg,c_bool),&
                                                         logical(opt_if_present_arg,c_bool),&
                                                         async_arg)
    else
      call gpufortrt_update_device_c_impl(hostptr,&
                                                   num_bytes,&
                                                   logical(opt_if_arg,c_bool),&
                                                   logical(opt_if_present_arg,c_bool))
    endif
  end subroutine

  subroutine gpufortrt_update_device0_l1(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(1,c_size_t),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device1_l1(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device2_l1(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device3_l1(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device4_l1(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device5_l1(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device6_l1(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device7_l1(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device0_l4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(4,c_size_t),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device1_l4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device2_l4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device3_l4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device4_l4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device5_l4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device6_l4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device7_l4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device0_ch1(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(1,c_size_t),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device1_ch1(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device2_ch1(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device3_ch1(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device4_ch1(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device5_ch1(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device6_ch1(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device7_ch1(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device0_i1(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(1,c_size_t),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device1_i1(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device2_i1(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device3_i1(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device4_i1(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device5_i1(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device6_i1(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device7_i1(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(1,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device0_i2(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(2,c_size_t),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device1_i2(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device2_i2(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device3_i2(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device4_i2(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device5_i2(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device6_i2(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device7_i2(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(2,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device0_i4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(4,c_size_t),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device1_i4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device2_i4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device3_i4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device4_i4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device5_i4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device6_i4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device7_i4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device0_i8(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(8,c_size_t),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device1_i8(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device2_i8(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device3_i8(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device4_i8(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device5_i8(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device6_i8(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device7_i8(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device0_r4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(4,c_size_t),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device1_r4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device2_r4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device3_r4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device4_r4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device5_r4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device6_r4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device7_r4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device0_r8(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(8,c_size_t),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device1_r8(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device2_r8(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device3_r8(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device4_r8(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device5_r8(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device6_r8(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device7_r8(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device0_c4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(2*4,c_size_t),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device1_c4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device2_c4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device3_c4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device4_c4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device5_c4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device6_c4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device7_c4(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(2*4,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device0_c8(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(2*8,c_size_t),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device1_c8(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device2_c8(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device3_c8(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device4_c8(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device5_c8(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device6_c8(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine

  subroutine gpufortrt_update_device7_c8(hostptr,if_arg,if_present_arg,async_arg)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    logical,intent(in),optional :: if_arg, if_present_arg
    integer(gpufortrt_handle_kind),intent(in),optional :: async_arg
    !
    call gpufortrt_update_device_b(c_loc(hostptr),int(2*8,c_size_t)*size(hostptr),if_arg,if_present_arg,async_arg)
  end subroutine



  subroutine gpufortrt_use_device0_l1(resultptr,hostptr,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    logical(c_bool),target,intent(in) :: hostptr
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    logical(c_bool),pointer,intent(inout) :: resultptr
    !
    type(c_ptr) :: tmp_cptr
    !
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr)
  end subroutine

  subroutine gpufortrt_use_device1_l1(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(*)
    integer,intent(in),optional :: sizes(1), lbounds(1)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    logical(c_bool),pointer,intent(inout) :: resultptr(:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(1), opt_lbounds(1)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device2_l1(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(1:1,*)
    integer,intent(in),optional :: sizes(2), lbounds(2)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    logical(c_bool),pointer,intent(inout) :: resultptr(:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(2), opt_lbounds(2)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device3_l1(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(1:1,1:1,*)
    integer,intent(in),optional :: sizes(3), lbounds(3)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    logical(c_bool),pointer,intent(inout) :: resultptr(:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(3), opt_lbounds(3)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device4_l1(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(4), lbounds(4)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    logical(c_bool),pointer,intent(inout) :: resultptr(:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(4), opt_lbounds(4)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device5_l1(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(5), lbounds(5)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    logical(c_bool),pointer,intent(inout) :: resultptr(:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(5), opt_lbounds(5)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device6_l1(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(6), lbounds(6)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    logical(c_bool),pointer,intent(inout) :: resultptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(6), opt_lbounds(6)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):,&
      opt_lbounds(6):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device7_l1(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(7), lbounds(7)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    logical(c_bool),pointer,intent(inout) :: resultptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(7), opt_lbounds(7)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):,&
      opt_lbounds(6):,&
      opt_lbounds(7):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device0_l4(resultptr,hostptr,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    logical,target,intent(in) :: hostptr
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    logical,pointer,intent(inout) :: resultptr
    !
    type(c_ptr) :: tmp_cptr
    !
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr)
  end subroutine

  subroutine gpufortrt_use_device1_l4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    logical,target,intent(in) :: hostptr(*)
    integer,intent(in),optional :: sizes(1), lbounds(1)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    logical,pointer,intent(inout) :: resultptr(:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(1), opt_lbounds(1)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device2_l4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    logical,target,intent(in) :: hostptr(1:1,*)
    integer,intent(in),optional :: sizes(2), lbounds(2)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    logical,pointer,intent(inout) :: resultptr(:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(2), opt_lbounds(2)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device3_l4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    logical,target,intent(in) :: hostptr(1:1,1:1,*)
    integer,intent(in),optional :: sizes(3), lbounds(3)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    logical,pointer,intent(inout) :: resultptr(:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(3), opt_lbounds(3)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device4_l4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    logical,target,intent(in) :: hostptr(1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(4), lbounds(4)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    logical,pointer,intent(inout) :: resultptr(:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(4), opt_lbounds(4)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device5_l4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    logical,target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(5), lbounds(5)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    logical,pointer,intent(inout) :: resultptr(:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(5), opt_lbounds(5)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device6_l4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    logical,target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(6), lbounds(6)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    logical,pointer,intent(inout) :: resultptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(6), opt_lbounds(6)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):,&
      opt_lbounds(6):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device7_l4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    logical,target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(7), lbounds(7)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    logical,pointer,intent(inout) :: resultptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(7), opt_lbounds(7)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):,&
      opt_lbounds(6):,&
      opt_lbounds(7):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device0_ch1(resultptr,hostptr,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    character(c_char),target,intent(in) :: hostptr
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    character(c_char),pointer,intent(inout) :: resultptr
    !
    type(c_ptr) :: tmp_cptr
    !
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr)
  end subroutine

  subroutine gpufortrt_use_device1_ch1(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    character(c_char),target,intent(in) :: hostptr(*)
    integer,intent(in),optional :: sizes(1), lbounds(1)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    character(c_char),pointer,intent(inout) :: resultptr(:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(1), opt_lbounds(1)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device2_ch1(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    character(c_char),target,intent(in) :: hostptr(1:1,*)
    integer,intent(in),optional :: sizes(2), lbounds(2)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    character(c_char),pointer,intent(inout) :: resultptr(:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(2), opt_lbounds(2)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device3_ch1(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    character(c_char),target,intent(in) :: hostptr(1:1,1:1,*)
    integer,intent(in),optional :: sizes(3), lbounds(3)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    character(c_char),pointer,intent(inout) :: resultptr(:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(3), opt_lbounds(3)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device4_ch1(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    character(c_char),target,intent(in) :: hostptr(1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(4), lbounds(4)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    character(c_char),pointer,intent(inout) :: resultptr(:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(4), opt_lbounds(4)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device5_ch1(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    character(c_char),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(5), lbounds(5)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    character(c_char),pointer,intent(inout) :: resultptr(:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(5), opt_lbounds(5)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device6_ch1(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    character(c_char),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(6), lbounds(6)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    character(c_char),pointer,intent(inout) :: resultptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(6), opt_lbounds(6)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):,&
      opt_lbounds(6):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device7_ch1(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    character(c_char),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(7), lbounds(7)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    character(c_char),pointer,intent(inout) :: resultptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(7), opt_lbounds(7)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):,&
      opt_lbounds(6):,&
      opt_lbounds(7):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device0_i1(resultptr,hostptr,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_int8_t),pointer,intent(inout) :: resultptr
    !
    type(c_ptr) :: tmp_cptr
    !
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr)
  end subroutine

  subroutine gpufortrt_use_device1_i1(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(*)
    integer,intent(in),optional :: sizes(1), lbounds(1)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_int8_t),pointer,intent(inout) :: resultptr(:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(1), opt_lbounds(1)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device2_i1(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(1:1,*)
    integer,intent(in),optional :: sizes(2), lbounds(2)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_int8_t),pointer,intent(inout) :: resultptr(:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(2), opt_lbounds(2)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device3_i1(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(1:1,1:1,*)
    integer,intent(in),optional :: sizes(3), lbounds(3)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_int8_t),pointer,intent(inout) :: resultptr(:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(3), opt_lbounds(3)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device4_i1(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(4), lbounds(4)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_int8_t),pointer,intent(inout) :: resultptr(:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(4), opt_lbounds(4)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device5_i1(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(5), lbounds(5)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_int8_t),pointer,intent(inout) :: resultptr(:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(5), opt_lbounds(5)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device6_i1(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(6), lbounds(6)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_int8_t),pointer,intent(inout) :: resultptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(6), opt_lbounds(6)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):,&
      opt_lbounds(6):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device7_i1(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(7), lbounds(7)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_int8_t),pointer,intent(inout) :: resultptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(7), opt_lbounds(7)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):,&
      opt_lbounds(6):,&
      opt_lbounds(7):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device0_i2(resultptr,hostptr,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_short),target,intent(in) :: hostptr
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_short),pointer,intent(inout) :: resultptr
    !
    type(c_ptr) :: tmp_cptr
    !
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr)
  end subroutine

  subroutine gpufortrt_use_device1_i2(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_short),target,intent(in) :: hostptr(*)
    integer,intent(in),optional :: sizes(1), lbounds(1)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_short),pointer,intent(inout) :: resultptr(:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(1), opt_lbounds(1)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device2_i2(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_short),target,intent(in) :: hostptr(1:1,*)
    integer,intent(in),optional :: sizes(2), lbounds(2)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_short),pointer,intent(inout) :: resultptr(:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(2), opt_lbounds(2)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device3_i2(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_short),target,intent(in) :: hostptr(1:1,1:1,*)
    integer,intent(in),optional :: sizes(3), lbounds(3)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_short),pointer,intent(inout) :: resultptr(:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(3), opt_lbounds(3)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device4_i2(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_short),target,intent(in) :: hostptr(1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(4), lbounds(4)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_short),pointer,intent(inout) :: resultptr(:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(4), opt_lbounds(4)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device5_i2(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_short),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(5), lbounds(5)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_short),pointer,intent(inout) :: resultptr(:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(5), opt_lbounds(5)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device6_i2(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_short),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(6), lbounds(6)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_short),pointer,intent(inout) :: resultptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(6), opt_lbounds(6)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):,&
      opt_lbounds(6):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device7_i2(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_short),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(7), lbounds(7)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_short),pointer,intent(inout) :: resultptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(7), opt_lbounds(7)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):,&
      opt_lbounds(6):,&
      opt_lbounds(7):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device0_i4(resultptr,hostptr,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_int),target,intent(in) :: hostptr
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_int),pointer,intent(inout) :: resultptr
    !
    type(c_ptr) :: tmp_cptr
    !
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr)
  end subroutine

  subroutine gpufortrt_use_device1_i4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_int),target,intent(in) :: hostptr(*)
    integer,intent(in),optional :: sizes(1), lbounds(1)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_int),pointer,intent(inout) :: resultptr(:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(1), opt_lbounds(1)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device2_i4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_int),target,intent(in) :: hostptr(1:1,*)
    integer,intent(in),optional :: sizes(2), lbounds(2)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_int),pointer,intent(inout) :: resultptr(:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(2), opt_lbounds(2)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device3_i4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_int),target,intent(in) :: hostptr(1:1,1:1,*)
    integer,intent(in),optional :: sizes(3), lbounds(3)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_int),pointer,intent(inout) :: resultptr(:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(3), opt_lbounds(3)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device4_i4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_int),target,intent(in) :: hostptr(1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(4), lbounds(4)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_int),pointer,intent(inout) :: resultptr(:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(4), opt_lbounds(4)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device5_i4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_int),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(5), lbounds(5)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_int),pointer,intent(inout) :: resultptr(:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(5), opt_lbounds(5)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device6_i4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_int),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(6), lbounds(6)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_int),pointer,intent(inout) :: resultptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(6), opt_lbounds(6)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):,&
      opt_lbounds(6):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device7_i4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_int),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(7), lbounds(7)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_int),pointer,intent(inout) :: resultptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(7), opt_lbounds(7)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):,&
      opt_lbounds(6):,&
      opt_lbounds(7):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device0_i8(resultptr,hostptr,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_long),target,intent(in) :: hostptr
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_long),pointer,intent(inout) :: resultptr
    !
    type(c_ptr) :: tmp_cptr
    !
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr)
  end subroutine

  subroutine gpufortrt_use_device1_i8(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_long),target,intent(in) :: hostptr(*)
    integer,intent(in),optional :: sizes(1), lbounds(1)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_long),pointer,intent(inout) :: resultptr(:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(1), opt_lbounds(1)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device2_i8(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_long),target,intent(in) :: hostptr(1:1,*)
    integer,intent(in),optional :: sizes(2), lbounds(2)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_long),pointer,intent(inout) :: resultptr(:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(2), opt_lbounds(2)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device3_i8(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_long),target,intent(in) :: hostptr(1:1,1:1,*)
    integer,intent(in),optional :: sizes(3), lbounds(3)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_long),pointer,intent(inout) :: resultptr(:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(3), opt_lbounds(3)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device4_i8(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_long),target,intent(in) :: hostptr(1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(4), lbounds(4)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_long),pointer,intent(inout) :: resultptr(:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(4), opt_lbounds(4)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device5_i8(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_long),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(5), lbounds(5)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_long),pointer,intent(inout) :: resultptr(:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(5), opt_lbounds(5)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device6_i8(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_long),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(6), lbounds(6)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_long),pointer,intent(inout) :: resultptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(6), opt_lbounds(6)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):,&
      opt_lbounds(6):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device7_i8(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    integer(c_long),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(7), lbounds(7)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    integer(c_long),pointer,intent(inout) :: resultptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(7), opt_lbounds(7)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):,&
      opt_lbounds(6):,&
      opt_lbounds(7):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device0_r4(resultptr,hostptr,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    real(c_float),target,intent(in) :: hostptr
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    real(c_float),pointer,intent(inout) :: resultptr
    !
    type(c_ptr) :: tmp_cptr
    !
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr)
  end subroutine

  subroutine gpufortrt_use_device1_r4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    real(c_float),target,intent(in) :: hostptr(*)
    integer,intent(in),optional :: sizes(1), lbounds(1)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    real(c_float),pointer,intent(inout) :: resultptr(:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(1), opt_lbounds(1)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device2_r4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    real(c_float),target,intent(in) :: hostptr(1:1,*)
    integer,intent(in),optional :: sizes(2), lbounds(2)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    real(c_float),pointer,intent(inout) :: resultptr(:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(2), opt_lbounds(2)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device3_r4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    real(c_float),target,intent(in) :: hostptr(1:1,1:1,*)
    integer,intent(in),optional :: sizes(3), lbounds(3)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    real(c_float),pointer,intent(inout) :: resultptr(:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(3), opt_lbounds(3)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device4_r4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    real(c_float),target,intent(in) :: hostptr(1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(4), lbounds(4)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    real(c_float),pointer,intent(inout) :: resultptr(:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(4), opt_lbounds(4)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device5_r4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    real(c_float),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(5), lbounds(5)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    real(c_float),pointer,intent(inout) :: resultptr(:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(5), opt_lbounds(5)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device6_r4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    real(c_float),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(6), lbounds(6)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    real(c_float),pointer,intent(inout) :: resultptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(6), opt_lbounds(6)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):,&
      opt_lbounds(6):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device7_r4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    real(c_float),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(7), lbounds(7)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    real(c_float),pointer,intent(inout) :: resultptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(7), opt_lbounds(7)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):,&
      opt_lbounds(6):,&
      opt_lbounds(7):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device0_r8(resultptr,hostptr,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    real(c_double),target,intent(in) :: hostptr
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    real(c_double),pointer,intent(inout) :: resultptr
    !
    type(c_ptr) :: tmp_cptr
    !
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr)
  end subroutine

  subroutine gpufortrt_use_device1_r8(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    real(c_double),target,intent(in) :: hostptr(*)
    integer,intent(in),optional :: sizes(1), lbounds(1)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    real(c_double),pointer,intent(inout) :: resultptr(:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(1), opt_lbounds(1)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device2_r8(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    real(c_double),target,intent(in) :: hostptr(1:1,*)
    integer,intent(in),optional :: sizes(2), lbounds(2)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    real(c_double),pointer,intent(inout) :: resultptr(:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(2), opt_lbounds(2)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device3_r8(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    real(c_double),target,intent(in) :: hostptr(1:1,1:1,*)
    integer,intent(in),optional :: sizes(3), lbounds(3)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    real(c_double),pointer,intent(inout) :: resultptr(:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(3), opt_lbounds(3)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device4_r8(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    real(c_double),target,intent(in) :: hostptr(1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(4), lbounds(4)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    real(c_double),pointer,intent(inout) :: resultptr(:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(4), opt_lbounds(4)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device5_r8(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    real(c_double),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(5), lbounds(5)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    real(c_double),pointer,intent(inout) :: resultptr(:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(5), opt_lbounds(5)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device6_r8(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    real(c_double),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(6), lbounds(6)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    real(c_double),pointer,intent(inout) :: resultptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(6), opt_lbounds(6)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):,&
      opt_lbounds(6):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device7_r8(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    real(c_double),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(7), lbounds(7)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    real(c_double),pointer,intent(inout) :: resultptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(7), opt_lbounds(7)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):,&
      opt_lbounds(6):,&
      opt_lbounds(7):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device0_c4(resultptr,hostptr,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    complex(c_float_complex),pointer,intent(inout) :: resultptr
    !
    type(c_ptr) :: tmp_cptr
    !
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr)
  end subroutine

  subroutine gpufortrt_use_device1_c4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(*)
    integer,intent(in),optional :: sizes(1), lbounds(1)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    complex(c_float_complex),pointer,intent(inout) :: resultptr(:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(1), opt_lbounds(1)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device2_c4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(1:1,*)
    integer,intent(in),optional :: sizes(2), lbounds(2)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    complex(c_float_complex),pointer,intent(inout) :: resultptr(:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(2), opt_lbounds(2)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device3_c4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(1:1,1:1,*)
    integer,intent(in),optional :: sizes(3), lbounds(3)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    complex(c_float_complex),pointer,intent(inout) :: resultptr(:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(3), opt_lbounds(3)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device4_c4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(4), lbounds(4)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    complex(c_float_complex),pointer,intent(inout) :: resultptr(:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(4), opt_lbounds(4)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device5_c4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(5), lbounds(5)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    complex(c_float_complex),pointer,intent(inout) :: resultptr(:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(5), opt_lbounds(5)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device6_c4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(6), lbounds(6)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    complex(c_float_complex),pointer,intent(inout) :: resultptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(6), opt_lbounds(6)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):,&
      opt_lbounds(6):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device7_c4(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(7), lbounds(7)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    complex(c_float_complex),pointer,intent(inout) :: resultptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(7), opt_lbounds(7)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):,&
      opt_lbounds(6):,&
      opt_lbounds(7):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device0_c8(resultptr,hostptr,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    complex(c_double_complex),pointer,intent(inout) :: resultptr
    !
    type(c_ptr) :: tmp_cptr
    !
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr)
  end subroutine

  subroutine gpufortrt_use_device1_c8(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(*)
    integer,intent(in),optional :: sizes(1), lbounds(1)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    complex(c_double_complex),pointer,intent(inout) :: resultptr(:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(1), opt_lbounds(1)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device2_c8(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(1:1,*)
    integer,intent(in),optional :: sizes(2), lbounds(2)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    complex(c_double_complex),pointer,intent(inout) :: resultptr(:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(2), opt_lbounds(2)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device3_c8(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(1:1,1:1,*)
    integer,intent(in),optional :: sizes(3), lbounds(3)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    complex(c_double_complex),pointer,intent(inout) :: resultptr(:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(3), opt_lbounds(3)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device4_c8(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(4), lbounds(4)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    complex(c_double_complex),pointer,intent(inout) :: resultptr(:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(4), opt_lbounds(4)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device5_c8(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(5), lbounds(5)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    complex(c_double_complex),pointer,intent(inout) :: resultptr(:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(5), opt_lbounds(5)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device6_c8(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(6), lbounds(6)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    complex(c_double_complex),pointer,intent(inout) :: resultptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(6), opt_lbounds(6)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):,&
      opt_lbounds(6):)&
        => resultptr
  end subroutine

  subroutine gpufortrt_use_device7_c8(resultptr,hostptr,sizes,lbounds,if_arg,if_present_arg)
    use iso_c_binding
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(1:1,1:1,1:1,1:1,1:1,1:1,*)
    integer,intent(in),optional :: sizes(7), lbounds(7)
    logical,intent(in),optional :: if_arg, if_present_arg
    !
    complex(c_double_complex),pointer,intent(inout) :: resultptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: tmp_cptr
    integer :: opt_sizes(7), opt_lbounds(7)
    !
    opt_sizes = 1
    opt_lbounds = 1  
    if ( present(sizes) ) opt_sizes = sizes
    if ( present(lbounds) ) opt_lbounds = lbounds
    tmp_cptr = gpufortrt_use_device_b(c_loc(hostptr),if_arg,if_present_arg)
    call c_f_pointer(tmp_cptr,resultptr,shape=opt_sizes)
    resultptr(&
      opt_lbounds(1):,&
      opt_lbounds(2):,&
      opt_lbounds(3):,&
      opt_lbounds(4):,&
      opt_lbounds(5):,&
      opt_lbounds(6):,&
      opt_lbounds(7):)&
        => resultptr
  end subroutine



  function gpufortrt_deviceptr0_l1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr1_l1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr2_l1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr3_l1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr4_l1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr5_l1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr6_l1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr7_l1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical(c_bool),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr0_l4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr1_l4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr2_l4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr3_l4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr4_l4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr5_l4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr6_l4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr7_l4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    logical,target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr0_ch1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr1_ch1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr2_ch1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr3_ch1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr4_ch1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr5_ch1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr6_ch1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr7_ch1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    character(c_char),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr0_i1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr1_i1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr2_i1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr3_i1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr4_i1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr5_i1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr6_i1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr7_i1(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int8_t),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr0_i2(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr1_i2(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr2_i2(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr3_i2(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr4_i2(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr5_i2(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr6_i2(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr7_i2(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_short),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr0_i4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr1_i4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr2_i4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr3_i4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr4_i4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr5_i4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr6_i4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr7_i4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_int),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr0_i8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr1_i8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr2_i8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr3_i8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr4_i8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr5_i8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr6_i8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr7_i8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    integer(c_long),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr0_r4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr1_r4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr2_r4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr3_r4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr4_r4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr5_r4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr6_r4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr7_r4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_float),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr0_r8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr1_r8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr2_r8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr3_r8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr4_r8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr5_r8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr6_r8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr7_r8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    real(c_double),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr0_c4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr1_c4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr2_c4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr3_c4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr4_c4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr5_c4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr6_c4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr7_c4(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_float_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr0_c8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr1_c8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr2_c8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr3_c8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr4_c8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr5_c8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr6_c8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function

  function gpufortrt_deviceptr7_c8(hostptr) result(deviceptr)
    use iso_c_binding
    use gpufortrt_types
    implicit none
    complex(c_double_complex),target,intent(in) :: hostptr(:,:,:,:,:,:,:)
    !
    type(c_ptr) :: deviceptr
    !
    deviceptr = gpufortrt_deviceptr_b(c_loc(hostptr))
  end function


end module