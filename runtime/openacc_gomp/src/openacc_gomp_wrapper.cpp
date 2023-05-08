// SPDX-License-Identifier: MIT                                                
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include <stddef.h>
#include <stdlib.h> 
#include <iostream>

extern "C" {
  void* acc_deviceptr (void* hostptr);
  // prototypes of oacc-parallel.c functions
  void GOACC_data_start (int device, size_t mapnum,void **hostaddrs, size_t *sizes, unsigned short *kinds);
  void GOACC_enter_exit_data (int device, size_t mapnum,
           void **hostaddrs, size_t *sizes, unsigned short *kinds,
           int async, int num_waits, ...);

  // interfaces between Fortran code (no variadics, no unsigned datypes to oacc-parallel.c functions

  void GOACC_data_start_wrapper (int device, size_t mapnum,
      void **hostaddrs, size_t *sizes, /*unsigned short*/int *kinds) {
    unsigned short _kinds[mapnum];
    for (unsigned int i=0; i<mapnum; i++) {
      _kinds[i] = (unsigned short) kinds[i];
    }
    if ( getenv("GOMP_DEBUG") != NULL ) {
      std::cout << "GOACC_data_start_wrapper" << "(";
      std::cout << "mapnum="<<mapnum;
      std::cout << ",hostaddrs=" << hostaddrs;
      std::cout << ",hostaddrs[:]=[";
      for (unsigned int i=0; i<mapnum; i++) { if ( i>0 ) {  std::cout << ","; }; std::cout << hostaddrs[i]; }
      std::cout << "]";
      std::cout << ",sizes=" << sizes;
      std::cout << ",sizes[:]=[";
      for (unsigned int i=0; i<mapnum; i++) { if ( i>0 ) {  std::cout << ","; }; std::cout << sizes[i]; }
      std::cout << "]";
      std::cout << ",kinds=" << kinds;
      std::cout << ",kinds[:]=[";
      for (unsigned int i=0; i<mapnum; i++) { if ( i>0 ) {  std::cout << ","; }; std::cout << kinds[i]; }
      std::cout << "]";
      std::cout << ")" << std::endl;
    }
    
    GOACC_data_start(device,mapnum,hostaddrs,sizes,_kinds);

    if ( getenv("GOMP_DEBUG") != NULL ) {
      std::cout << "GOACC_data_start_wrapper" << "(";
      std::cout << "device variables[:]=[";
      for (unsigned int i=0; i<mapnum; i++) { if ( i>0 ) {  std::cout << ","; }; std::cout << acc_deviceptr(hostaddrs[i]); }
      std::cout << "])" << std::endl;
    }
  }
  
  void GOACC_enter_exit_data_wrapper (int device, size_t mapnum,
      void **hostaddrs, size_t *sizes, /*unsigned short*/int *kinds, int async, int num_waits,int* waits) {
    unsigned short _kinds[mapnum];
    for (unsigned int i=0; i<mapnum; i++) {
      _kinds[i] = (unsigned short) kinds[i];
    }
    if ( getenv("GOMP_DEBUG") != NULL ) {
      std::cout << "GOACC_enter_exit_data_wrapper" << "(";
      std::cout << "mapnum="<<mapnum;
      std::cout << ",hostaddrs=" << hostaddrs;
      std::cout << ",hostaddrs[:]=[";
      for (unsigned int i=0; i<mapnum; i++) { if ( i>0 ) {  std::cout << ","; }; std::cout << hostaddrs[i]; }
      std::cout << "]";
      std::cout << ",sizes=" << sizes;
      std::cout << ",sizes[:]=[";
      for (unsigned int i=0; i<mapnum; i++) { if ( i>0 ) {  std::cout << ","; }; std::cout << sizes[i]; }
      std::cout << "]";
      std::cout << ",kinds=" << kinds;
      std::cout << ",kinds[:]=[";
      for (unsigned int i=0; i<mapnum; i++) { if ( i>0 ) {  std::cout << ","; }; std::cout << kinds[i]; }
      std::cout << "]";
      std::cout << ",async="<<async;
      std::cout << ",num_waits="<<num_waits;
      std::cout << ",waits[:]=[";
      for (unsigned int i=0; i<num_waits; i++) { if ( i>0 ) {  std::cout << ","; }; std::cout << waits[i]; }
      std::cout << "]";
      std::cout << ")" << std::endl;
    } 
    // unroll array in order to call function with variadic argument list 
    switch (num_waits) {
        /*
        jinja2 template:

{% for i in range(0,10+1) %}      case {{i}}:
        GOACC_enter_exit_data(device,mapnum,hostaddrs,sizes,_kinds,async,num_waits{% for k in range(0,i) %},waits[{{k}}]{% endfor %});
        break;
{% endfor %}

        use, e.g., here: https://j2live.ttl255.com/
        */
      case 0:   
        GOACC_enter_exit_data(device,mapnum,hostaddrs,sizes,_kinds,async,num_waits);
        break;
      case 1:   
        GOACC_enter_exit_data(device,mapnum,hostaddrs,sizes,_kinds,async,num_waits,waits[0]);
        break;
      case 2:   
        GOACC_enter_exit_data(device,mapnum,hostaddrs,sizes,_kinds,async,num_waits,waits[0],waits[1]);
        break;
      case 3:   
        GOACC_enter_exit_data(device,mapnum,hostaddrs,sizes,_kinds,async,num_waits,waits[0],waits[1],waits[2]);
        break;
      case 4:   
        GOACC_enter_exit_data(device,mapnum,hostaddrs,sizes,_kinds,async,num_waits,waits[0],waits[1],waits[2],waits[3]);
        break;
      case 5:   
        GOACC_enter_exit_data(device,mapnum,hostaddrs,sizes,_kinds,async,num_waits,waits[0],waits[1],waits[2],waits[3],waits[4]);
        break;
      case 6:   
        GOACC_enter_exit_data(device,mapnum,hostaddrs,sizes,_kinds,async,num_waits,waits[0],waits[1],waits[2],waits[3],waits[4],waits[5]);
        break;
      case 7:   
        GOACC_enter_exit_data(device,mapnum,hostaddrs,sizes,_kinds,async,num_waits,waits[0],waits[1],waits[2],waits[3],waits[4],waits[5],waits[6]);
        break;
      case 8:   
        GOACC_enter_exit_data(device,mapnum,hostaddrs,sizes,_kinds,async,num_waits,waits[0],waits[1],waits[2],waits[3],waits[4],waits[5],waits[6],waits[7]);
        break;
      case 9:   
        GOACC_enter_exit_data(device,mapnum,hostaddrs,sizes,_kinds,async,num_waits,waits[0],waits[1],waits[2],waits[3],waits[4],waits[5],waits[6],waits[7],waits[8]);
        break;
      case 10:   
        GOACC_enter_exit_data(device,mapnum,hostaddrs,sizes,_kinds,async,num_waits,waits[0],waits[1],waits[2],waits[3],waits[4],waits[5],waits[6],waits[7],waits[8],waits[9]);
        break;
      default:
        // TODO print warning
        break;
    }

    if ( getenv("GOMP_DEBUG") != NULL ) {
      std::cout << "GOACC_enter_exit_data_wrapper" << "(";
      std::cout << "device variables[:]=[";
      for (unsigned int i=0; i<mapnum; i++) { if ( i>0 ) {  std::cout << ","; }; std::cout << acc_deviceptr(hostaddrs[i]); }
      std::cout << "])" << std::endl;
    }
  }

  // auxiliary

  void print_cptr(void* ptr)  {
    std::cout << ptr;
  }  
}
