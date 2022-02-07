#!/usr/bin/env python3
# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.

# executes __init__.py scripts from top to bottom
print(">import pkg.subpkg.mod:")
import pkg.subpkg.mod
# output:
# pkg.__init__.py
# pkg.subpgk.__init__.py
# pkg.subpkg.mod

# executes __init__.py scripts from top to bottom
# BUT only the ones that have not been executed yet
print(">import pkg.mod:")
import pkg.mod


