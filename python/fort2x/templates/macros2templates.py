#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import addtoplevelpath
import fort2x.templategen

import os
import re

macro_files = [
  "filegen.macros.cpp",
  "filegen.macros.f03",
  "derived_type.macros.f03",
  "derived_type.macros.cpp",
  ]
macro_filters = [
  re.compile("|".join(r"""render_interface_module
render_used_modules
render_c_file""".split("\n"))),
  re.compile("|".join(r"""render_derived_type_copy_array_member_routines
render_derived_type_destroy_array_member_routines
render_derived_type_init_array_member_routines
render_derived_type_copy_scalars_routines
render_derived_type_destroy_routines
render_derived_types
render_derived_type_size_bytes_routines""".split("\n"))),
  ]
    
root_dir  = os.path.abspath(os.path.join(__file__,".."))
generator = fort2x.templategen.TemplateGenerator(root_dir,
                                                 macro_files,
                                                 macro_filters)
generator.verbose = False
generator.convert_macros_to_templates()
