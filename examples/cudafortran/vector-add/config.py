# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.

#fort2x.hip.opts.emit_debug_code = false

#fort2x.hip.opts.prettify_emitted_fortran_code = false

def blockdims(kernelname,dims):
    return [512]

def launchbounds(kernelname):
    print(kernelname)
    return "512,2"

#fort2x.hip.opts.get_block_dims = blockdims

#fort2x.hip.opts.get_launch_bounds = launchbounds

opts.prettify_modified_translation_source = true

opts.log_dir="log"

scanner.opts.loop_kernel_default_launcher="hip_ps"