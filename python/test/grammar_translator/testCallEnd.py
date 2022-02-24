#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import sys
import test
from gpufort import translator
import addtoplevelpath
from gpufort import grammar

testdata = """
1 )
a_d )
psi_d )
2 * lda, ps_d, 1, 1.D0, psi_d, 1 )
spsi_d )
a_d )
1, spsi_d )
1, 1, spsi_d )
lda, ps_d, 1, 1, spsi_d )
lda, ps_d )
lda, ps_d, 1, 1, spsi_d, 1 )
2 * lda, ps_d, 1, 1, spsi_d, 1 )
2 * lda, ps_d, 1, 1.D0, spsi_d, 1 )
""".strip("\n").strip(" ").strip("\n").splitlines()

test.run(
   expression     = translator.call_end,
   testdata       = testdata,
   tag            = "call_end",
   raiseException = True
)