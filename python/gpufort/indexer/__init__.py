# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from .indexer import *
from . import scope
from . import indexertypes
from . import props
from . import intrinsics

#from . import opts # might not work; might create new opts module that differs from the package-local opts module
from .indexer import opts # imports package-local opts module