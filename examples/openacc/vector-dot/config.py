# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
LOG_DIR="log"

fort2hip.CLANG_FORMAT_STYLE = "\"{BasedOnStyle: llvm, ColumnLimit: 140}\""

fort2hip.EMITTED_DEBUG_CODE = False

fort2hip.PRETTIFY_MODIFIED_TRANSLATION_SOURCE = True

scanner.LOOP_KERNEL_DEFAULT_LAUNCHER= "auto"