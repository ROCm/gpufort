#!/usr/bin/env python3
# SPDX-License-Identifier: MIT                                                
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.

# executes __init__.py scripts from top to bottom
print(">import pkg.subpkg.mod:")
import pkg
print(pkg.subpkg)
print(pkg.subpkg.mod)

print(pkg.MyClass.myclassvar)
pkg.MyClass.myclassvar = 2
print(pkg.MyClass.myclassvar)
print(pkg.mod.MyClass.myclassvar)

pkg.mod.MyClass.myclassvar = 3
print(pkg.MyClass.myclassvar)
print(pkg.mod.MyClass.myclassvar)

