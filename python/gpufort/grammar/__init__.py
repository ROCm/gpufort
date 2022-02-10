# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import os

GRAMMAR_DIR  = os.path.dirname(os.path.abspath(__file__))
GRAMMAR_PATH = os.path.join(GRAMMAR_DIR,"grammar.py")

from .grammar import *