# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from gpufort import util
from .. import render
from . import opts

import os


def generate_gpufort_headers(output_dir):
    """Create the header files that all GPUFORT HIP kernels rely on."""
    util.logging.log_enter_function(opts.log_prefix,"generate_gpufort_headers",\
      {"output_dir": output_dir})

    gpufort_header_file_path = os.path.join(output_dir, "gpufort.h")
    render.render_gpufort_header_file(gpufort_header_file_path)
    msg = "created gpufort main header: ".ljust(40) + gpufort_header_file_path
    util.logging.log_info(opts.log_prefix, "generate_gpufort_headers", msg)

    gpufort_reduction_header_file_path = os.path.join(output_dir,
                                                      "gpufort_reduction.h")
    render.render_gpufort_reduction_header_file(
        gpufort_reduction_header_file_path)
    msg = "created gpufort reductions header file: ".ljust(
        40) + gpufort_reduction_header_file_path
    util.logging.log_info(opts.log_prefix, "generate_gpufort_headers", msg)

    # gpufort arrays
    gpufort_array_context = {
        "max_rank": opts.max_dim,
        "datatypes": opts.datatypes
    }
    gpufort_array_header_file_path = os.path.join(output_dir,
                                                  "gpufort_array.h")
    render.render_gpufort_array_header_file(gpufort_array_header_file_path,\
      context=gpufort_array_context)
    msg = "created gpufort arrays header file: ".ljust(
        40) + gpufort_array_header_file_path
    util.logging.log_info(opts.log_prefix, "generate_gpufort_headers", msg)

    util.logging.log_info(opts.log_prefix, "generate_gpufort_headers", msg)


def generate_gpufort_sources(output_dir):
    """Create the source files that all GPUFORT HIP kernels rely on."""

    util.logging.log_enter_function(opts.log_prefix,"generate_gpufort_sources",\
      {"output_dir": output_dir})

    # gpufort arrays
    gpufort_array_context = {
        "max_rank": opts.max_dim,
        "datatypes": opts.datatypes,
    }
    gpufort_array_source_file_path = os.path.join(output_dir,
                                                  "gpufort_array.hip.cpp")
    render.render_gpufort_array_source_file(gpufort_array_source_file_path,\
      context=gpufort_array_context)
    msg = "".join([
        "created gpufort arrays C++ source file: ".ljust(40),
        gpufort_array_source_file_path
    ])
    util.logging.log_info(opts.log_prefix, "generate_gpufort_sources", msg)

    gpufort_array_fortran_interfaces_module_path = os.path.join(
        output_dir, "gpufort_array.f03")
    render.render_gpufort_array_fortran_interfaces_file(gpufort_array_fortran_interfaces_module_path,\
                                                               context=gpufort_array_context)
    msg = "created gpufort arrays Fortran interface module: ".ljust(
        40) + gpufort_array_fortran_interfaces_module_path
    util.logging.log_info(opts.log_prefix, "generate_gpufort_sources", msg)

    util.logging.log_leave_function(opts.log_prefix,
                                    "generate_gpufort_sources")
