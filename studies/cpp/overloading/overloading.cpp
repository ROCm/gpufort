// SPDX-License-Identifier: MIT
// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
#include <iostream>

void foo(float a) {
  std::cout << "float" << std::endl;
}

void foo(double b) {
  std::cout << "double" << std::endl;
}

//void foo(float& a) { // no ption for literals
//  std::cout << "float reference" << std::endl;
//}
//
//void foo(double& b) { // no option for literals
//  std::cout << "double reference" << std::endl;
//}

int main(int argc, char** argv) {
   double f = 2.00f;
   double d = 2.00;
	
   foo(1.00f);
   foo(1.00);
   
   foo(f);
   foo(d);
}