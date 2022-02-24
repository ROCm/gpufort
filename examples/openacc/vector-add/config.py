# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
scanner.DESTINATION_DIALECT="hipgpufort"
scanner.LOOP_KERNEL_DEFAULT_LAUNCHER="auto"

fort2hip.CLANG_FORMAT_STYLE="\"{BasedOnStyle: llvm, ColumnLimit: 140, BinPackArguments: false, BinPackParameters: false}\""

LOG_DIR="log"
#LOG_LEVEL="debug"

#utils.logging.opts.verbose = True

ENABLE_PROFILING=False