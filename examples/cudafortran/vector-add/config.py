# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
fort2x.hip.opts.clang_format_style="\"{BasedOnStyle: llvm, ColumnLimit: 140, BinPackArguments: false, BinPackParameters: false}\""

#fort2hip.EMIT_DEBUG_CODE = False

#fort2hip.PRETTIFY_EMITTED_FORTRAN_CODE = False

def myBlockDims(kernelName,dims):
    return [512]

def myLaunchBounds(kernelName):
    print(kernelName)
    return "512,2"

#fort2hip.GET_BLOCK_DIMS = myBlockDims

#fort2hip.GET_LAUNCH_BOUNDS = myLaunchBounds

opts.prettify_modified_translation_source = True

opts.log_dir="log"

scanner.opts.loop_kernel_default_launcher="hip_ps"
