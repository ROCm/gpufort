// SPDX-License-Identifier: MIT
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#include <iomanip>
#include <sstream>

#ifndef AUXILIARY_H
#define AUXILIARY_H

#define WS_FILE 40
#define WS_LINE 4

#define LOG_INFO(level,msg) \
    if ( level <= gpufortrt::internal::LOG_LEVEL ) { \
        std::stringstream ss_prefix; \
        std::string file(__FILE__); \
        file.replace(0,6,""); \
        std::string line = std::to_string(__LINE__); \
        ss_prefix << std::noskipws << std::left << std::setw(WS_FILE) << file << ":" \
                  << std::setw(WS_LINE) << line << ":"; \
        std::stringstream ss_msg; \
        ss_msg << msg; \
        gpufortrt::internal::log_info(level,ss_prefix.str(),ss_msg.str()); }

#define LOG_ERROR(msg) { \
        std::stringstream ss_prefix; \
        std::string file(__FILE__); \
        file.replace(0,6,""); \
        std::string line = std::to_string(__LINE__); \
        ss_prefix << std::noskipws << std::left << std::setw(WS_FILE) << file << ":" \
                  << std::setw(WS_LINE) << line << ":"; \
        std::stringstream ss_msg; \
        ss_msg << msg; \
        gpufortrt::internal::log_error(ss_prefix.str(),ss_msg.str()); }

#define HIP_CHECK(error) \
  { \
    if(error != 0){ \
        LOG_ERROR("HIP runtime call returned error status: " << error) \
    } \
  }

namespace gpufortrt {
  namespace internal {
    /** Set `variable` from environment variable with identifier `env_var`.*/
    void set_from_environment(int& variable,const char* env_var); 
    void set_from_environment(size_t& variable,const char* env_var);
    void set_from_environment(double& variable,const char* env_var);

    extern int LOG_LEVEL; //< The global log level, defaults to 0.

    /** Prints info output to error output stream if log `level`
     * is less than or equal to global `LOG_LEVEL`.
     * \param[in] level the log level of this info message.
     * \param[in] msg The message.
     *
     * \note Use the LOG_INFO macro to use the << operator in the `msg` argument.
     */
    void log_info(const int level,const std::string& prefix,const std::string& msg);
    /** Prints error output to error stream and terminates the application.
     * \note Use the LOG_INFO macro to use the << operator in the `msg` argument.
     */
    void log_error(const std::string& prefix,const std::string& msg);
  }
}
#endif // AUXILIARY_H
