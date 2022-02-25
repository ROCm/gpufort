#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import addtoplevelpath
from gpufort.fort2x import templategen

import os
import re

macro_files = [
    "hip_implementation.macros.cpp",
    "interface_module.macros.f03",
]
macro_filters = [
    re.compile("|".join(r"""render_hip_kernel\b
render_hip_device_routine
render_hip_device_routine_comment
render_hip_launcher
render_launcher\b
render_cpu_routine
render_cpu_launcher""".splitlines()))
]

root_dir = os.path.abspath(os.path.join(__file__, ".."))
generator = templategen.TemplateGenerator(root_dir, macro_files, macro_filters)
generator.verbose = False
generator.convert_macros_to_templates()
