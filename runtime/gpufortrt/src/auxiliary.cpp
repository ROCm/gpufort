// SPDX-License-Identifier: MIT
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
int gpufortrt::internal::LOG_LEVEL = 0;

void gpufortrt::internal::set_from_environment(int& variable,const char* env_var) {
  char* val = getenv(env_var);
  if ( val != nullptr ) {
    variable = std::stoi(val);
  }
}

void gpufortrt::internal::set_from_environment(size_t& variable,const char* env_var) {
  char* val = getenv(env_var);
  if ( val != nullptr ) {
    variable = std::stol(val);
  }
}

void gpufortrt::internal::set_from_environment(double& variable,const char* env_var) {
  char* val = getenv(env_var);
  if ( val != nullptr ) {
    variable = std::stod(val);
  }
}
    
void gpufortrt::internal::log_info(int level,std::string& msg) {
  std::cerr << "[gpufortrt][" << level << "] " << msg << std::endl;
}

void gpufortrt::internal::log_error(std::string& msg) {
  std::cerr << "[gpufortrt] ERROR: " << msg << std::endl;
  std::terminate();
}
