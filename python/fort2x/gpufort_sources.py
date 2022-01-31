# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import os,sys

import addtoplevelpath
import utils.logging

import fort2x.fort2x
import fort2x.render

def generate_gpufort_headers(output_dir):
    """Create the header files that all GPUFORT HIP kernels rely on."""
    utils.logging.log_enter_function(fort2x.fort2x.LOG_PREFIX,"generate_gpufort_headers",\
      {"output_dir": output_dir})
    
    gpufort_header_file_path = os.path.join(output_dir,"gpufort.h")
    fort2x.render.render_gpufort_header_file(gpufort_header_file_path)
    msg = "created gpufort main header: ".ljust(40) + gpufort_header_file_path
    utils.logging.log_info(fort2x.fort2x.LOG_PREFIX,"generate_gpufort_headers",msg)
    
    gpufort_reduction_header_file_path = os.path.join(output_dir, "gpufort_reduction.h")
    fort2x.render.render_gpufort_reduction_header_file(gpufort_reduction_header_file_path)
    msg = "created gpufort reductions header file: ".ljust(40) + gpufort_reduction_header_file_path
    utils.logging.log_info(fort2x.fort2x.LOG_PREFIX,"generate_gpufort_headers",msg)
    
    # gpufort arrays
    gpufort_array_context={
      "max_rank":fort2x.fort2x.GPUFORT_HEADERS_MAX_DIM,
      "datatypes":fort2x.fort2x.GPUFORT_HEADERS_DATATYPES
    }
    gpufort_array_header_file_path = os.path.join(output_dir,"gpufort_array.h")
    fort2x.render.render_gpufort_array_header_file(gpufort_array_header_file_path,\
      context=gpufort_array_context)
    msg = "created gpufort arrays header file: ".ljust(40) + gpufort_array_header_file_path
    utils.logging.log_info(fort2x.fort2x.LOG_PREFIX,"generate_gpufort_headers",msg)
    
    utils.logging.log_info(fort2x.fort2x.LOG_PREFIX,"generate_gpufort_headers",msg)

def generate_gpufort_sources(output_dir):
    """Create the source files that all GPUFORT HIP kernels rely on."""
    
    utils.logging.log_enter_function(fort2x.fort2x.LOG_PREFIX,"generate_gpufort_sources",\
      {"output_dir": output_dir})
    
    # gpufort arrays
    gpufort_array_context = {
      "max_rank" : fort2x.fort2x.GPUFORT_HEADERS_MAX_DIM,
      "datatypes" : fort2x.fort2x.GPUFORT_HEADERS_DATATYPES,
      }
    gpufort_array_source_file_path = os.path.join(output_dir,"gpufort_array.hip.cpp")
    fort2x.render.render_gpufort_array_source_file(gpufort_array_source_file_path,\
      context=gpufort_array_context)
    msg = "".join(["created gpufort arrays C++ source file: ".ljust(40),gpufort_array_source_file_path])
    utils.logging.log_info(fort2x.fort2x.LOG_PREFIX,"generate_gpufort_sources",msg)
    
    gpufort_array_fortran_interfaces_module_path = os.path.join(output_dir,
                                                                "gpufort_array.f03")
    fort2x.render.render_gpufort_array_fortran_interfaces_file(gpufort_array_fortran_interfaces_module_path,\
                                                               context=gpufort_array_context)
    msg = "created gpufort arrays Fortran interface module: ".ljust(40) + gpufort_array_fortran_interfaces_module_path
    utils.logging.log_info(fort2x.fort2x.LOG_PREFIX,"generate_gpufort_sources",msg)

    utils.logging.log_leave_function(fort2x.fort2x.LOG_PREFIX,"generate_gpufort_sources")
