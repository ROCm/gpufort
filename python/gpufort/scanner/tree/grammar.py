# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
CASELESS = False
from gpufort.grammar import GRAMMAR_DIR, GRAMMAR_PATH # GRAMMAR_DIR must be set as global var

exec(open(GRAMMAR_PATH).read())
