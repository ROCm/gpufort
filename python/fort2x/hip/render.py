# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
import os
import pprint

import jinja2

import addtoplevelpath
import utils.logging
        
from fort2x.render import generate_code

fort2hip_dir = os.path.dirname(__file__)
exec(open(os.path.join(fort2hip_dir,"templates","render.py.in")).read())
