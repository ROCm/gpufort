// SPDX-License-Identifier: MIT
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include <iostream>

struct B{
  double d1;
  int i2;
};

struct A{
  int i1;
  int& i2;
  B& b3;
};

extern "C" {
  void print_struct_(A& a) {
    std::cout << a.i1 << std::endl;
    std::cout << a.i2 << std::endl; 
    std::cout << a.b3.d1 << std::endl; 
    std::cout << a.b3.i2 << std::endl; 
  }
}
