# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
fort2hip.CLANG_FORMAT_STYLE="\"{BasedOnStyle: llvm, ColumnLimit: 140, BinPackArguments: false, BinPackParameters: false}\""

LOG_DIR="./log"

LOG_LEVEL="warning"

utils.logging.opts.verbose = False

#utils.logging.LOG_FILTER = "scanner"

utils.logging.SKIP_CREATE_GPUFORT_MODULE_FILES = True