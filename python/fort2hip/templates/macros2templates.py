#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import addtoplevelpath
import fort2hip.templategen

import os
import re

macro_files = [
  "hip_implementation.macros.cpp",
  "gpufort_array.macros.h",
  "interface_module.macros.f03",
  "gpufort_array.macros.f03",
  ]
macro_filters = [
  re.compile("|".join(r"""render_derived_type_copy_array_member_routines
render_derived_type_copy_scalars_routines
render_derived_types
render_derived_type_size_bytes_routines
render_(begin|end)_kernel
render_hip_kernel\b
render_hip_kernel_comment
render_hip_launcher
render_hip_device_routine
render_launcher\b
render_cpu_routine
render_cpu_launcher
render_interface_module
render_c_file""".split("\n"))),
  re.compile("|".join(r"""render_gpufort_array_c_bindings
render_gpufort_array_data_access_interfaces
render_gpufort_array_data_access_routines
render_gpufort_array_init_routines
render_gpufort_array_copy_to_buffer_routines
render_gpufort_array_wrap_routines
render_gpufort_array_interfaces""".split("\n")))
  ]
    
root_dir  = os.path.abspath(os.path.join(__file__,".."))
generator = fort2hip.templategen.TemplateGenerator(root_dir,
                                          macro_files,
                                          macro_filters)
generator.verbose = True
generator.convert_macros_to_templates()
