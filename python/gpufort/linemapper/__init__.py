# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from .linemapper import *
from . import grammar

#from . import opts # might not work; might create new opts module that differs from the package-local opts module
from .linemapper import opts # imports package-local opts module
