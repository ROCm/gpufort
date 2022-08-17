! SPDX-License-Identifier: MIT                                                
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
program test_c_bindings
  use iso_c_binding
  use gpufortrt_c_bindings
  implicit none
  !
  real(4),target :: array(10,10)
  type(c_ptr) :: base = c_null_ptr
  type(c_ptr) :: sub  = c_null_ptr
  type(c_ptr) :: tmp  = c_null_ptr
  integer(c_size_t) :: offset_bytes
  logical :: fits
  !
  base = c_loc(array) 
  sub  = c_loc(array(:,2)) ! offset_bytes should be (2-1)*10*4 bytes from base pointer
  CALL print_cptr(base); Print *, ""
  CALL print_cptr(sub); Print *, ""
  fits = is_subarray(base,size(array)*4_8,sub,0_8,offset_bytes)
  print *, "offset_bytes=", offset_bytes 
  tmp = inc_cptr(base,offset_bytes)
  CALL print_cptr(tmp); Print *, ""
  
  if ( c_associated(sub,tmp) ) then
    print *, "test_c_bindings PASSED!"
  else
    print *, "test_c_bindings FAILED!"
  endif
end program