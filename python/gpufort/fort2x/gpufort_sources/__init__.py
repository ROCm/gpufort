# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from .gpufort_sources import *

#from . import opts # might not work; might create new opts module that differs from the package-local opts module
from .gpufort_sources import opts # imports package-local opts module
