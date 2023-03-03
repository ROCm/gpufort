# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
import pyparsing

from gpufort import util

from .. import tree
from .. import optvals

class CufKernelDoInfo:
    def __init__(self):
        self.num_loops = optvals.OptionalSingleValue()
        self.grid = optvals.OptionalSingleValue()
        self.block = optvals.OptionalSingleValue()
        self.sharedmem = optvals.OptionalSingleValue()
        self.stream = optvals.OptionalSingleValue()
        self.reduction = optvals.OptionalDictValue()

def _analyze_directive_action(ttnode,parents,result):
    if isinstance(ttnode,tree.TTCufKernelDoArgNumLoops):
        result.num_loops.value = ttnode.expr
    elif isinstance(ttnode,tree.TTCufKernelDoArgGrid):
        if ttnode.expr != "*":
            result.grid.value = ttnode.expr
    elif isinstance(ttnode,tree.TTCufKernelDoArgBlock):
        if ttnode.expr != "*":
            result.block.value = ttnode.expr
    elif isinstance(ttnode,tree.TTCufKernelDoArgSharedmem):
        result.sharedmem.value = ttnode.expr
    elif isinstance(ttnode,tree.TTCufKernelDoArgStream):
        result.stream.value = ttnode.expr

def analyze_directive(ttcufkerneldo):
    result = CufKernelDoInfo()
    #tree.traversals.traverse(
    #    ttcufkerneldo,
    #    _analyze_directive_action,
    #    tree.traversals.no_action,
    #    tree.traversals.no_crit,
    #    result)
    return result
