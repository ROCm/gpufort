#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import addtoplevelpath
import fort2x.templategen

import os
import re

macro_files = [
  "filegen.macros.cpp",
  "filegen.macros.f03"
  ]
macro_filters = [
  re.compile("|".join(r"""render_interface_module
render_c_file""".split("\n")))
  ]
    
root_dir  = os.path.abspath(os.path.join(__file__,".."))
generator = fort2x.templategen.TemplateGenerator(root_dir,
                                                 macro_files,
                                                 macro_filters)
generator.verbose = True
generator.convert_macros_to_templates()
