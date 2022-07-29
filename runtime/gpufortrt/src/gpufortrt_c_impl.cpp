// SPDX-License-Identifier: MIT                                                
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "gpufortrt_c_impl.h"

#include <iostream>

std ostream& operator<<(std::ostream& os, gpufortrt::map_kind_t map_kind);
{
    switch(ce)
    {
       case gpufortrt::map_kind_t::Dec_struct_refs: os << "dec_struct_refs"; break;;
       case gpufortrt::map_kind_t::Undefined      : os << "undefined"; break;;
       case gpufortrt::map_kind_t::Present        : os << "present"; break;;
       case gpufortrt::map_kind_t::Delete         : os << "delete"; break;;
       case gpufortrt::map_kind_t::Create         : os << "create"; break;;
       case gpufortrt::map_kind_t::No_create      : os << "no_create"; break;;
       case gpufortrt::map_kind_t::Copyin         : os << "copyin"; break;;
       case gpufortrt::map_kind_t::Copyout        : os << "copyout"; break;;
       case gpufortrt::map_kind_t::Copy           : os << "copy"; break;;
       default: os.setstate(std ios_base failbit);;
    }
    return os;
}

void gpufortrt::record_t::to_string(std::ostream& os)
  os << std::unitbuf
     << "global id:"          << id        
     << ", hostptr:"          << hostptr  
     << ", deviceptr:"        << deviceptr
     << ", initialized:"      << initialized 
     << ", used:"             << used 
     << ", released:"         << released 
     << ", num_bytes:"        << num_bytes 
     << ", struct_refs:"      << struct_refs  
     << ", dyn_refs:"         << dyn_refs  
     << ", map_kind:"         << 
     static_cast<gpufortr::map_kind_t>(map_kind) << std::endl;
}

bool gpufortrt::record_t::to_string(std::ostream& os)
  os << std::unitbuf
     << "global id:"          << id        
     << ", hostptr:"          << hostptr  
     << ", deviceptr:"        << deviceptr
     << ", initialized:"      << initialized 
     << ", used:"             << used 
     << ", released:"         << released 
     << ", num_bytes:"        << num_bytes 
     << ", struct_refs:"      << struct_refs  
     << ", dyn_refs:"         << dyn_refs  
     << ", map_kind:"         << 
     static_cast<gpufortr::map_kind_t>(map_kind) << std::endl;
}

// C bindings

extern "C" {
  char* inc_cptr(char* ptr, size_t offset_bytes)  {
    return ptr + offset_bytes;
  }  

  bool is_subarray(char* arr, size_t bytes, char* other, size_t bytes_other,size_t* relative_offset)  {
    *relative_offset = other - arr;
    return (*relative_offset >= 0) && ((*relative_offset+bytes_other) < bytes);    
  } 
  
  void print_cptr(void* ptr)  {
    std::cout << std::unitbuf
              << ptr << std::flush;
  }

  /**
   * \param[in] error_id Error enum, see gpufortrt_c_impl.h
   */
  void print_error(int error_id) {
  }

  void print_subarray(void* base_hostptr, void* base_deviceptr, size_t base_num_bytes,
                                void* section_hostptr, void* section_deviceptr, size_t section_num_bytes) {
    std::cout << std::unitbuf
              << "[section_deviceptr: " << section_deviceptr 
              << ", section_hostptr: " << section_hostptr
              << ", section_num_bytes: " << section_num_bytes
              << ", base_deviceptr: " << base_deviceptr
              << ", base_hostptr: " << base_hostptr
              << ", base_num_bytes: " << base_num_bytes << "]" << std::endl;
  }

  void print_record(int id,
                    bool initialized,
                    bool used,
                    bool released,
                    void* hostptr,
                    void* deviceptr,
                    size_t num_bytes,
                    int struct_refs,
                    int dyn_refs,
                    int creational_event) {
    std::cout << std::unitbuf
              << "global id:"          << id        
              << ", hostptr:"          << hostptr  
              << ", deviceptr:"        << deviceptr
              << ", initialized:"      << initialized 
              << ", used:"             << used 
              << ", released:"         << released 
              << ", num_bytes:"        << num_bytes 
              << ", struct_refs:"      << struct_refs  
              << ", dyn_refs:"         << dyn_refs  
              << ", map_kind:" << static_cast<gpufortr::map_kind_t>(map_kind) << std::endl;
  }
}
