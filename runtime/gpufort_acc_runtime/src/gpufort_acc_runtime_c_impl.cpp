// SPDX-License-Identifier: MIT                                                
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include "gpufort_acc_runtime_c_impl.h"

#include <iostream>

extern "C" {
  char* inc_cptr(char*& ptr, size_t& offset_bytes)  {
    return ptr + offset_bytes;
  }  

  bool is_subarray(char*& arr, size_t& bytes, char*& other, size_t& bytes_other,size_t*& relative_offset)  {
    *relative_offset = other - arr;
    return (other >= arr) && (*relative_offset < bytes) && ((other+bytes_other) <= arr+bytes);    
  } 
  
  void print_cptr(void*& ptr)  {
    std::cout << ptr;
  }  

  void print_record(int& id,
                    bool& initialized,
                    bool& used,
                    bool& released,
                    void*& hostptr,
                    void*& deviceptr,
                    size_t& num_bytes,
                    int& struct_refs,
                    int& dyn_refs,
                    int& creational_event) {
    std::cout << "global id:"          << id        
              << ", hostptr:"          << hostptr   
              << ", deviceptr:"        << deviceptr 
              << ", initialized:"      << initialized 
              << ", used:"             << used 
              << ", released:"         << released 
              << ", num_bytes:"        << num_bytes 
              << ", struct_refs:"      << struct_refs  
              << ", dyn_refs:"         << dyn_refs  
              << ", creational_event:" << static_cast<record_creational_event>(creational_event) << std::endl;
  }
}
