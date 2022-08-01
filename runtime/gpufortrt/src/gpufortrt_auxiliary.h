// SPDX-License-Identifier: MIT
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
namespace gpufortrt {
  namespace auxiliary {
    void set_from_environment(int& variable,const char* env_var); 
    void set_from_environment(size_t& variable,const char* env_var);
    void set_from_environment(double& variable,const char* env_var);
  }
}