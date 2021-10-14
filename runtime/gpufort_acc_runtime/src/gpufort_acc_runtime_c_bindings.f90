! SPDX-License-Identifier: MIT                                                
! Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
module gpufort_acc_runtime_c_bindings

  interface
    
    function inc_cptr(ptr,offset_bytes) bind(c,name="inc_cptr")
      use iso_c_binding
      implicit none
      type(c_ptr),intent(in),value :: ptr
      integer(c_size_t),value      :: offset_bytes
      !
      type(c_ptr) :: inc_cptr
    end function
    
    function is_subarray(arr,bytes,other,bytes_other,relative_offset_bytes) bind(c,name="is_subarray")
      use iso_c_binding
      implicit none
      type(c_ptr),intent(in),value       :: arr,other
      integer(c_size_t),intent(in),value :: bytes,bytes_other
      integer(c_size_t),intent(inout)    :: relative_offset_bytes
      !
      logical(c_bool) :: is_subarray
    end function

    subroutine print_cptr(ptr) bind(c,name="print_cptr")
      use iso_c_binding
      implicit none
      type(c_ptr),intent(in),value :: ptr
    end subroutine

    subroutine print_record(id,initialized,hostptr,deviceptr,num_bytes,num_refs,region,creational_event) bind(c,name="print_record")
      use iso_c_binding
      implicit none
      integer,intent(in),value           :: id
      logical(c_bool),intent(in)         :: initialized
      type(c_ptr),intent(in),value       :: hostptr
      type(c_ptr),intent(in),value       :: deviceptr
      integer(c_size_t),intent(in),value   :: num_bytes
      integer,intent(in),value           :: num_refs
      integer,intent(in),value           :: region
      integer,intent(in),value           :: creational_event
    end subroutine

  end interface
end module