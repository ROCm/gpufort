// SPDX-License-Identifier: MIT
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#define LOG_INFO(level,msg) \
    if ( level <= gpufortrt::internal::LOG_LEVEL ) { std::stringstream ss; ss << msg; log_info(level,ss.str()); }

#define LOG_ERROR(msg) \
    { std::stringstream ss; ss << msg; log_error(ss.str()); }

namespace gpufortrt {
  namespace internal {
    /** Set `variable` from environment variable with identifier `env_var`.*/
    void set_from_environment(int& variable,const char* env_var); 
    void set_from_environment(size_t& variable,const char* env_var);
    void set_from_environment(double& variable,const char* env_var);

    int LOG_LEVEL; //< The global log level, defaults to 0.

    /** Prints info output to error output stream if log `level`
     * is less than or equal to global `LOG_LEVEL`.
     * \param[in] level the log level of this info message.
     * \param[in] msg The message.
     *
     * \note Use the LOG_INFO macro to use the << operator in the `msg` argument.
     */
    void log_info(const int level,const std::string& msg);
    /** Prints error output to error stream and terminates the application.
     * \note Use the LOG_INFO macro to use the << operator in the `msg` argument.
     */
    void log_error(const std::string& msg);
  }
}
