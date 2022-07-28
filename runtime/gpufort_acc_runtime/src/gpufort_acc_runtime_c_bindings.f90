! SPDX-License-Identifier: MIT                                                
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
module gpufort_acc_runtime_c_bindings

  interface
    
    function inc_cptr(ptr,offset_bytes) bind(c,name="inc_cptr")
      use iso_c_binding
      implicit none
      type(c_ptr),intent(in) :: ptr
      integer(c_size_t)      :: offset_bytes
      !
      type(c_ptr) :: inc_cptr
    end function
    
    function is_subarray(arr,bytes,other,bytes_other,relative_offset_bytes) bind(c,name="is_subarray")
      use iso_c_binding
      implicit none
      type(c_ptr),intent(in)          :: arr,other
      integer(c_size_t),intent(in)    :: bytes,bytes_other
      integer(c_size_t),intent(inout) :: relative_offset_bytes
      !
      logical(c_bool) :: is_subarray
    end function

    subroutine print_cptr(ptr) bind(c,name="print_cptr")
      use iso_c_binding
      implicit none
      type(c_ptr),intent(in) :: ptr
    end subroutine

    subroutine print_record(id,&
                            initialized,&
                            used,& 
                            released,&
                            hostptr,&
                            deviceptr,&
                            num_bytes,&
                            struct_refs,&
                            dyn_refs,&
                            map_kind) bind(c,name="print_record")
      use iso_c_binding
      implicit none
      integer,intent(in)           :: id
      logical(c_bool),intent(in)   :: initialized
      logical(c_bool),intent(in)   :: used 
      logical(c_bool),intent(in)   :: released
      type(c_ptr),intent(in)       :: hostptr
      type(c_ptr),intent(in)       :: deviceptr
      integer(c_size_t),intent(in) :: num_bytes
      integer,intent(in)           :: struct_refs
      integer,intent(in)           :: dyn_refs
      integer,intent(in)           :: map_kind
    end subroutine

  end interface
end module