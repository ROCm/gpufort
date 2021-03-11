#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 GPUFORT Advanced Micro Devices, Inc. All rights reserved.
import addtoplevelpath
import sys
import test
import grammar as grammar

print(grammar.use.parseString("use kinds, only: dp, sp => sp2"))
