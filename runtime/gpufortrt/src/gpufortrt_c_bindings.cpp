#include "gpufortrt_core.h"

// C bindings
//extern "C" {
//  char* inc_cptr(char* ptr, size_t offset_bytes)  {
//    return ptr + offset_bytes;
//  }  
//
//  bool is_subarray(char* arr, size_t bytes, char* other, size_t bytes_other,size_t* relative_offset)  {
//    *relative_offset = other - arr;
//    return (*relative_offset >= 0) && ((*relative_offset+bytes_other) < bytes);    
//  } 
//  
//  void print_cptr(void* ptr)  {
//    std::cout << std::unitbuf
//              << ptr << std::flush;
//  }
//
//  void print_subarray(void* base_hostptr, void* base_deviceptr, size_t base_num_bytes,
//                                void* section_hostptr, void* section_deviceptr, size_t section_num_bytes) {
//    std::cout << std::unitbuf
//              << "[section_deviceptr: " << section_deviceptr 
//              << ", section_hostptr: " << section_hostptr
//              << ", section_num_bytes: " << section_num_bytes
//              << ", base_deviceptr: " << base_deviceptr
//              << ", base_hostptr: " << base_hostptr
//              << ", base_num_bytes: " << base_num_bytes << "]" << std::endl;
//  }
//
//  void print_record(gpufortrt::record_t& record) {
//    std::cout << record << std::endl;
//  }
//}
//

extern "C" {
  
} // extern "C"
